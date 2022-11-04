import numpy as np
import argparse
import os
import json
import torch.utils.data as data
import random
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from dataset import MiniDataset, EpisodeSampler
from util import accuracy, averageMeter, lr_decay
from model import Convnet, MLP, Hallucinator, w_Discriminator
import pdb

def worker_init_fn(worker_id):           
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def main():
    global save_dir, logger
    
    # fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # setup directory to save logfiles, checkpoints, and output csv
    save_dir = args.save_dir
    if args.phase == 'train' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # setup logger
    logger = None
    if args.phase == 'train':
        logger = open(os.path.join(save_dir, 'train.log'), 'a')
        logfile = os.path.join(save_dir, 'training_log.json')
        log = {'train': [], 'val': []}
        logger.write('{}\n'.format(args))
    
    # setup data loader for training images
    if args.phase == 'train':
        csv_file = os.path.join(args.data_root, 'train.csv')
        data_dir = os.path.join(args.data_root, 'train')
        dataset_train = MiniDataset(csv_file, data_dir)
        sampler_train = EpisodeSampler(args.n_way_train, args.n_shot, args.n_query, n_classes=64)
        train_loader = data.DataLoader(
            dataset_train, batch_size=args.n_way_train * (args.n_shot + args.n_query), num_workers=args.n_worker, \
            pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler_train
        )
        
        print('train: {}'.format(dataset_train.__len__()))
        logger.write('train: {}\n'.format(dataset_train.__len__()))
        
    # setup data loader for validation/testing images    
    csv_file = os.path.join(args.data_root, 'val.csv')
    data_dir = os.path.join(args.data_root, 'val')
    dataset_val = MiniDataset(csv_file, data_dir)
    
    print('val/test: {}'.format(dataset_val.__len__()))
    if args.phase == 'train':
        logger.write('val/test: {}\n'.format(dataset_val.__len__()))
    
    sampler_val = EpisodeSampler(args.n_way_test, args.n_shot, args.n_query, n_classes=16)
    val_loader = data.DataLoader(
        dataset_val, batch_size=args.n_way_test * (args.n_shot + args.n_query), num_workers=args.n_worker, \
        pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler_val
    )
    
    # setup model (& dist function)
    cnn = Convnet().cuda()
    gen = Hallucinator().cuda()
    D = w_Discriminator().cuda()
    model_params = list(cnn.parameters()) + list(gen.parameters())
    if args.phase == 'train':
        logger.write('{}\n'.format(cnn))
        logger.write('{}\n'.format(gen))
        logger.write('{}\n'.format(D))
    
    dist_func = None
    if args.dist_func == 'parametric':
        dist_func = MLP().cuda()
        model_params += list(dist_func.parameters())    
        if args.phase == 'train':
            logger.write('{}\n'.format(dist_func))
   
    # setup criterion
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # setup optimizer
    optimizer = torch.optim.RMSprop(model_params, lr=args.lr)
    opt_D = torch.optim.RMSprop(D.parameters(), lr=args.lr_d, weight_decay=1e-4)
    if args.phase == 'train':
        logger.write('{}\n'.format(optimizer))
        logger.write('{}\n'.format(opt_D))
    
    # load checkpoint
    start_ep = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        cnn.load_state_dict(checkpoint['cnn_state'])
        gen.load_state_dict(checkpoint['gen_state'])
        D.load_state_dict(checkpoint['D_state'])
        if checkpoint.get('mlp_state', None):
            dist_func.load_state_dict(checkpoint['mlp_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])
        opt_D.load_state_dict(checkpoint['opt_D_state'])
        start_ep = checkpoint['epoch']
        print("Loaded checkpoint '{}' (epoch: {})".format(args.checkpoint, start_ep))
        
        if args.phase == 'train':
            logger.write("Loaded checkpoint '{}' (epoch: {})\n".format(args.checkpoint, start_ep))
            if os.path.isfile(logfile):
                log = json.load(open(logfile, 'r'))

    if args.phase == 'train':
        
        # start training
        print('Start training from epoch {}'.format(start_ep))
        logger.write('Start training from epoch {}\n'.format(start_ep))
        for epoch in range(start_ep, args.epochs):
            
            loss_train, loss_c_train, loss_d_train, loss_g_train, acc_train = train(
                train_loader, cnn, gen, D, dist_func, optimizer, opt_D, epoch, criterion
            )
            log['train'].append([epoch + 1, loss_train, acc_train])
            
            if (epoch + 1) % args.val_ep == 0:
                with torch.no_grad():
                    loss_val, acc_val = val(val_loader, cnn, gen, dist_func, criterion, num_episode=20)
                log['val'].append([epoch + 1, loss_val, acc_val])
                
                # save checkpoint
                if args.dist_func == 'parametric':
                    state = {
                        'epoch': epoch + 1,
                        'cnn_state': cnn.state_dict(),
                        'gen_state': gen.state_dict(),
                        'D_state': D.state_dict(),
                        'mlp_state': dist_func.state_dict(),
                        'opt_state': optimizer.state_dict(),
                        'opt_D_state': opt_D.state_dict()
                    }
                else:
                    state = {
                        'epoch': epoch + 1,
                        'cnn_state': cnn.state_dict(),
                        'gen_state': gen.state_dict(),
                        'D_state': D.state_dict(),
                        'opt_state': optimizer.state_dict(),
                        'opt_D_state': opt_D.state_dict()
                    }
                checkpoint = os.path.join(save_dir, 'ep-{}.pkl'.format(epoch + 1))
                #torch.save(state, checkpoint)
                print('[Checkpoint] {} is saved.'.format(checkpoint))
                logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                json.dump(log, open(logfile, 'w'))
                
            if (epoch + 1) % args.step == 0:
                lr_decay(optimizer, decay_rate=args.gamma)
                lr_decay(opt_D, decay_rate=args.gamma)
        
        # save last model
        if args.dist_func == 'parametric':
            state = {
                'epoch': epoch + 1,
                'cnn_state': cnn.state_dict(),
                'gen_state': gen.state_dict(),
                'D_state': D.state_dict(),
                'mlp_state': dist_func.state_dict(),
                'opt_state': optimizer.state_dict(),
                'opt_D_state': opt_D.state_dict()
            }
        else:
            state = {
                'epoch': epoch + 1,
                'cnn_state': cnn.state_dict(),
                'gen_state': gen.state_dict(),
                'D_state': D.state_dict(),
                'opt_state': optimizer.state_dict(),
                'opt_D_state': opt_D.state_dict()
            }
        checkpoint = os.path.join(save_dir, 'last_checkpoint.pkl')
        torch.save(state, checkpoint)
        print('[Checkpoint] {} is saved.'.format(checkpoint))
        logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
        print('Training is done.')
        logger.write('Training is done.\n')
        logger.close()
        
    else:
        with torch.no_grad():
            loss_val, acc_val = val(val_loader, cnn, gen, dist_func, criterion, save_result=args.save_feat)
            
        print('Testing is done.')

            
def train(data_loader, cnn, gen, D, dist_func, optimizer, opt_D, epoch, criterion):

    losses = averageMeter()
    c_losses = averageMeter()
    d_losses = averageMeter()
    g_losses = averageMeter()
    acc = averageMeter()
    acc_d = averageMeter()
    
    # setup training mode
    cnn.train()
    gen.train()
    D.train()
    if dist_func is not None: # parametric
        dist_func.train()
    
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[1]

        """ Train D """
        D.zero_grad()
        
        for p in D.parameters():
            p.requires_grad = True
            p.data.clamp_(-0.01, 0.01)
            
        with torch.no_grad():
            # forward
            feat = cnn(image).detach()

            # split data into support and query data
            support_feat = feat[:args.n_way_train * args.n_shot, :]
            query_feat = feat[args.n_way_train * args.n_shot:, :]
        
            # hallucinate features
            # get seed features
            support_feat = support_feat.reshape(args.n_way_train, args.n_shot, -1)
            seed_feat = support_feat[:, 0, :].unsqueeze(dim=1).repeat(1, args.n_aug, 1)

            # generate noise
            noise = torch.randn([args.n_way_train, args.n_aug, feat.size(1)]).cuda()
        
            # generate hallucinated features and labels
            aug_feat = gen(seed_feat, noise).detach()
               
            # label (1: real, 0: fake)
            n_img = min(args.n_query, args.n_aug)
            r_img = query_feat.reshape(args.n_way_train, args.n_query, -1)[:, :n_img, :]
            r_img = r_img.reshape(args.n_way_train * n_img, -1)
            f_img = aug_feat[:, :n_img, :].reshape(args.n_way_train * n_img, -1)
            r_label = torch.ones([args.n_way_train * n_img, 1]).cuda()
            f_label = torch.zeros([args.n_way_train * n_img, 1]).cuda()
            
        # compute discriminator score
        r_logit = D(r_img.detach())
        f_logit = D(f_img.detach())
            
        # compute discriminator loss
        loss_d = f_logit.mean(0).view(1) - r_logit.mean(0).view(1)
        d_losses.update(loss_d.item(), n_img * 2)
        
        # compute discriminator accuracy
        with torch.no_grad():
            r_logit = torch.sigmoid(r_logit)
            f_logit = torch.sigmoid(f_logit)
            acc_1 = torch.ge(r_logit, 0.5).sum().float() / (args.n_way_train * n_img)
            acc_0 = torch.lt(f_logit, 0.5).sum().float() / (args.n_way_train * n_img)
            dacc = (acc_0 + acc_1) / 2
            acc_d.update(dacc.item(), args.n_way_train * n_img * 2)
            
        # backward
        loss_d.backward()
        opt_D.step()
        
        """ Train Prototypical Classifier and Hallucinator """
        optimizer.zero_grad()
        cnn.zero_grad()
        gen.zero_grad()
        
        for p in D.parameters():
            p.requires_grad = False
            
        # create the relative label (0 ~ N_way-1) for query data
        label_encoder = {target[i * args.n_shot]: i for i in range(args.n_way_train)}
        query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.n_way_train * args.n_shot:]])
        
        # forward
        feat = cnn(image)

        # split data into support and query data
        support_feat = feat[:args.n_way_train * args.n_shot, :]
        query_feat = feat[args.n_way_train * args.n_shot:, :]
        
        # hallucinate features
        # get seed features
        support_feat = support_feat.reshape(args.n_way_train, args.n_shot, -1)
        seed_feat = support_feat[:, 0, :].unsqueeze(dim=1).repeat(1, args.n_aug, 1)

        # generate noise
        noise = torch.randn([args.n_way_train, args.n_aug, feat.size(1)]).cuda()
        
        # generate hallucinated features and labels
        aug_feat = gen(seed_feat.detach(), noise)
        
        # compute class prototypes from the feature of support samples
        support_feat = torch.cat((support_feat, aug_feat), dim=1)
        proto = support_feat.mean(dim=1)
        
        # compute distance
        query_feat_repeat = query_feat.unsqueeze(dim=1).repeat(1, args.n_way_train, 1)
        proto_repeat = proto.unsqueeze(dim=0).repeat(args.n_way_train * args.n_query, 1, 1)
        if args.dist_func == 'euclidean':
            dist = torch.sum((proto_repeat - query_feat_repeat) ** 2, dim=2)
        elif args.dist_func == 'cosine':
            dist = -F.cosine_similarity(proto_repeat, query_feat_repeat, dim=2)
        else: # args.dist_func == 'parametric'
            dist = dist_func(proto_repeat, query_feat_repeat, dim=2)
        
        # compute classification loss
        loss_c = criterion(-dist, query_label)
        c_losses.update(loss_c.item(), image.size(0))

        # compute accuracy
        with torch.no_grad():
            prec = accuracy(-dist, query_label, topk=(1,))
            acc.update(prec[0].item(), image.size(0))

        # compute discriminator score
        f_logit = D(aug_feat.reshape(args.n_way_train * args.n_aug, -1))
        r_label = torch.ones([args.n_way_train * args.n_aug, 1]).cuda()
            
        # compute generator loss
        loss_g = -f_logit.mean(0).view(1)
        g_losses.update(loss_g.item(), args.n_way_train * args.n_aug)
        
        # compute total loss
        loss = loss_c + args.lamda * loss_g
        losses.update(loss.item(), image.size(0))
        
        # backward
        loss.backward()
        optimizer.step()
    
    # logging
    if (epoch + 1) % args.print_ep == 0:
        curr_lr = optimizer.param_groups[0]['lr']
        curr_lr_d = opt_D.param_groups[0]['lr']
        print('Epoch: [{}/{}] ' \
            'LR: [{:.6g}] ' \
            'LR_D: [{:.6g}] ' \
            'Loss {loss.avg:.4f} ' \
            'Loss_C {c_loss.avg:.4f} ' \
            'Loss_D {d_loss.avg:.4f} ' \
            'Loss_G {g_loss.avg:.4f} ' \
            'Acc {acc.avg:.4f} ' \
            'Acc_D {acc_d.avg:.4f}'.format(
                epoch + 1, args.epochs, curr_lr, curr_lr_d, loss=losses, c_loss=c_losses, d_loss=d_losses, \
                g_loss=g_losses, acc=acc, acc_d=acc_d
            )
        )
        logger.write('Epoch: [{}/{}] ' \
            'LR: [{:.6g}] ' \
            'LR_D: [{:.6g}] ' \
            'Loss {loss.avg:.4f} ' \
            'Loss_C {c_loss.avg:.4f} ' \
            'Loss_D {d_loss.avg:.4f} ' \
            'Loss_G {g_loss.avg:.4f} ' \
            'Acc {acc.avg:.4f} ' \
            'Acc_D {acc_d.avg:.4f}\n'.format(
                epoch + 1, args.epochs, curr_lr, curr_lr_d, loss=losses, c_loss=c_losses, d_loss=d_losses, \
                g_loss=g_losses, acc=acc, acc_d=acc_d
            )
        )
    
    return losses.avg, c_losses.avg, d_losses.avg, g_losses.avg, acc.avg

    
def val(data_loader, cnn, gen, dist_func, criterion, num_episode=600, save_result=False):

    losses = averageMeter()
    acc = averageMeter()
    
    if save_result:
        que_feat = []
        hal_feat = []
        que_label = []
        hal_label = []
    
    # setup evaluation mode
    cnn.eval()
    gen.eval()
    if dist_func is not None: # parametric
        dist_func.eval()
    
    for ep in range(num_episode):
        for (step, value) in enumerate(data_loader):

            image = value[0].cuda()
            target = value[1]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.n_shot]: i for i in range(args.n_way_test)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.n_way_test * args.n_shot:]])

            # forward
            feat = cnn(image)

            # split data into support and query data
            support_feat = feat[:args.n_way_test * args.n_shot, :]
            query_feat = feat[args.n_way_test * args.n_shot:, :]
            
            # hallucinate features
            # get seed features
            support_feat = support_feat.reshape(args.n_way_test, args.n_shot, -1)
            seed_feat = support_feat[:, 0, :].unsqueeze(dim=1).repeat(1, args.n_aug, 1)

            # generate noise
            noise = torch.randn([args.n_way_test, args.n_aug, feat.size(1)]).cuda()

            # generate hallucinated features and labels
            aug_feat = gen(seed_feat, noise)
            
            if save_result and ep == 0:
                que_feat.extend(query_feat.data.cpu().numpy().tolist())
                que_label.extend(query_label.data.cpu().numpy().tolist())
                hal_feat.extend(aug_feat.reshape(args.n_way_test * args.n_aug, -1).data.cpu().numpy().tolist())
                aug_label = torch.arange(args.n_way_test).unsqueeze(dim=1).repeat(1, args.n_aug)
                hal_label.extend(aug_label.reshape(args.n_way_test * args.n_aug).data.numpy().tolist())
            
            # compute class prototypes from the feature of support samples
            support_feat = torch.cat((support_feat, aug_feat), dim=1)
            proto = support_feat.mean(dim=1)

            # compute distance
            query_feat_repeat = query_feat.unsqueeze(dim=1).repeat(1, args.n_way_test, 1)
            proto_repeat = proto.unsqueeze(dim=0).repeat(args.n_way_test * args.n_query, 1, 1)
            if args.dist_func == 'euclidean':
                dist = torch.sum((proto_repeat - query_feat_repeat) ** 2, dim=2)
            elif args.dist_func == 'cosine':
                dist = -F.cosine_similarity(proto_repeat, query_feat_repeat, dim=2)
            else: # args.dist_func == 'parametric'
                dist = dist_func(proto_repeat, query_feat_repeat, dim=2)

            # compute loss
            loss = criterion(-dist, query_label)
            losses.update(loss.item(), image.size(0))

            # compute accuracy
            prec = accuracy(-dist, query_label, topk=(1,))
            acc.update(prec[0].item(), image.size(0))
        
    # logging
    mean, std = np.array(acc.values).mean(), np.array(acc.values).std()
    print('[Val] Loss {:.4f}\tAcc {:.3f} +- {:.3f}'.format(losses.avg, mean, 1.96 * std / (num_episode)**(1/2)))
    if args.phase == 'train':
        logger.write('[Val] Loss {:.4f}\tAcc {:.3f} +- {:.3f}\n'.format(losses.avg, mean, 1.96 * std / (num_episode)**(1/2)))
    
    if args.save_feat:
        np.save(os.path.join(save_dir, 'que_feat.npy'), np.asarray(que_feat))
        np.save(os.path.join(save_dir, 'que_label.npy'), np.asarray(que_label))
        np.save(os.path.join(save_dir, 'hal_feat.npy'), np.asarray(hal_feat))
        np.save(os.path.join(save_dir, 'hal_label.npy'), np.asarray(hal_label))
    
    return losses.avg, mean


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000, help='number of episodes to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate (default: 1e-3)')
    parser.add_argument('--lr_d', type=float, default=1e-6, help='base learning rate of discriminator (default: 1e-6)')
    parser.add_argument('--step', type=int, default=5000, help='learning rate decay step (default: 5000)')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate step gamma (default: 0.5)')
    parser.add_argument('--val_ep', type=int, default=200, help='validation period (default: 200)')
    parser.add_argument('--print_ep', type=int, default=20, help='validation period (default: 20)')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    parser.add_argument('--n_worker', type=int, default=3, help='number of workers (default: 3)')
    parser.add_argument('--n_way_train', type=int, default=30, help='number of classes in an episode (default: 30)')
    parser.add_argument('--n_way_test', type=int, default=5, help='number of classes for testing (default: 5)')
    parser.add_argument('--n_shot', type=int, default=1, help='number of support samples per class (default: 1)')
    parser.add_argument('--n_query', type=int, default=15, help='number of query samples per class (default: 15)')
    parser.add_argument('--n_aug', type=int, default=15, help='number of augmented samples per class (default: 15)')
    parser.add_argument('--lamda', type=float, default=1, help='hyperparameter for generator loss (default: 1)')
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained model')
    parser.add_argument('--save_dir', type=str, default='checkpoint/euclidean', help='directory to save logfile and checkpoint')
    parser.add_argument('--save_feat', type=bool, default=False, help='save features and corresponding labels for TSNE plot')
    parser.add_argument('--data_root', type=str, default='data', help='data root')
    parser.add_argument('--dist_func', type=str, default='euclidean', help='distance function (euclidean/cosine/parametric)')
    parser.add_argument('--phase', type=str, default='train', help='phase (train/test)')
    
    args = parser.parse_args()
    print(args)
    
    main()