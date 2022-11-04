import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

class MiniDataset(data.Dataset):
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

class EpisodeSampler(Sampler):
    def __init__(self, n_way, n_shot, n_query, n_classes=64, n_samples=600):
        self.n_way = n_way              # number of classes in an episode
        self.n_shot = n_shot            # number of support samples per class
        self.n_query = n_query          # number of query samples per class
        self.n_classes = n_classes
        self.n_samples = n_samples

    def __iter__(self):
        sampled_cls = torch.randperm(self.n_classes)[:self.n_way]
        support_idx = []
        query_idx = []
        for i in range(self.n_way):
            cls = sampled_cls[i]
            idx_cls = torch.randperm(self.n_samples)[:self.n_shot + self.n_query]
            idx = cls * self.n_samples + idx_cls
            support_idx.append(idx[:self.n_shot])
            query_idx.append(idx[self.n_shot:])
        return iter(torch.cat(support_idx + query_idx).tolist())

    def __len__(self):
        return self.n_way * (self.n_shot + self.n_query)

"""
import pdb
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
n_way, n_shot, n_query, n_episode = 5, 2, 10, 30
train_dataset = MiniDataset('data/train.csv', 'data/train')
train_loader = data.DataLoader(
        train_dataset, batch_size=n_way * (n_shot + n_query), num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=EpisodeSampler(n_way, n_shot, n_query))

for ep in range(n_episode):
    print(ep)
    for (step, value) in enumerate(train_loader):
        data = value[0].cuda()
        target = value[1]

        # split data into support and query data
        support_input = data[:n_way * n_shot, :, :, :]
        query_input = data[n_way * n_shot:, :, :, :]

        # create the relative label (0 ~ N_way-1) for query data
        label_encoder = {target[i * n_shot] : i for i in range(n_way)}
        query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[n_way * n_shot:]])

        pdb.set_trace()
"""
