import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F

import csv
import random
import numpy as np
import pandas as pd

from model import Convnet, MLP, Hallucinator
import pdb

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, cnn, gen, dist_func, data_loader):
    
    prediction_results = []
    with torch.no_grad():
        
        # setup evaluation mode
        cnn.eval()
        if gen is not None:
            gen.eval()
        if dist_func is not None: # parametric
            dist_func.eval()
        
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):

            # extract feature of support and query data
            feat = cnn(data.cuda())
            
            # split data into support and query data
            support_feat = feat[:args.N_way * args.N_shot, :]
            query_feat   = feat[args.N_way * args.N_shot:, :]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])
            
            support_feat = support_feat.reshape(args.N_way, args.N_shot, -1)
            
            if args.has_hallucinator: # P2 & P3
                # hallucinate features
                # get seed features
                seed_feat = support_feat[:, 0, :].unsqueeze(dim=1).repeat(1, args.N_aug, 1)

                # generate noise
                noise = torch.randn([args.N_way, args.N_aug, feat.size(1)]).cuda()

                # generate hallucinated features and labels
                aug_feat = gen(seed_feat, noise)
                
                # compute class prototypes from the feature of support samples
                support_feat = torch.cat((support_feat, aug_feat), dim=1)
            
            # compute class prototypes from the feature of support samples
            proto = support_feat.mean(dim=1)
                
            query_feat_repeat = query_feat.unsqueeze(dim=1).repeat(1, args.N_way, 1)
            proto_repeat = proto.unsqueeze(dim=0).repeat(args.N_way * args.N_query, 1, 1)
            if args.dist_func == 'euclidean':
                dist = torch.sum((proto_repeat - query_feat_repeat) ** 2, dim=2)
            elif args.dist_func == 'cosine':
                dist = -F.cosine_similarity(proto_repeat, query_feat_repeat, dim=2)
            else: # args.dist_func == 'parametric'
                dist = dist_func(proto_repeat, query_feat_repeat, dim=2)

            # classify the query data depending on the its distance with each prototype
            pred = torch.max(-dist, dim=1)[1].data.cpu().numpy().tolist()
            prediction_results.append(pred)

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--N_aug', default=10, type=int, help='N_aug (default: 10)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, default='data/val.csv', help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, default='data/val', help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, default='data/val_testcase.csv', help="Test case csv")
    parser.add_argument('--output_csv', type=str, default='val_testcase_pred.csv', help="Output filename")
    parser.add_argument('--has_hallucinator', type=bool, default=True, help='take data hallucination, default=True')
    parser.add_argument('--dist_func', type=str, default='euclidean', help='distance function (euclidean/cosine/parametric)')

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    print(args)
    
    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # setup model
    cnn = Convnet().cuda()
    gen = None
    if args.has_hallucinator:
        gen = Hallucinator().cuda()
    
    dist_func = None
    if args.dist_func == 'parametric':
        dist_func = MLP().cuda()
    
    # load checkpoint
    checkpoint = torch.load(args.load)
    cnn.load_state_dict(checkpoint['cnn_state'])
    if args.has_hallucinator:
        gen.load_state_dict(checkpoint['gen_state'])
    
    if dist_func is not None:
        dist_func.load_state_dict(checkpoint['mlp_state'])
    print("Loaded checkpoint '{}'".format(args.load))
    
    # get prediction results
    prediction_results = predict(args, cnn, gen, dist_func, test_loader)

    # output your prediction to csv
    with open(args.output_csv, 'w') as f:
        header = 'episode_id,' + ','.join(['query{}'.format(i) for i in range(len(prediction_results[0]))])
        f.write('{}\n'.format(header))
        for i in range(len(prediction_results)):
            x = prediction_results[i]
            data = '{},'.format(i) + ','.join([str(x[i]) for i in range(len(x))])
            f.write('{}\n'.format(data))
    print('Prediction results are saved to {}'.format(args.output_csv))
