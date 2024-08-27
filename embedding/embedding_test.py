'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import json
import numpy as np
import models
import datasets
import math

from BatchAverage import BatchCriterion
from utils import *
from tensorboardX import SummaryWriter

import CacheDataset

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--dataset', default='cifar',
                    help='dataset name: "cifar": cifar-10 datasetor "stl": stl-10 dataset]')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--log_dir', default='log/', type=str,
                    help='log save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str,
                    help='model save path')
parser.add_argument('--test_epoch', default=1, type=int,
                    metavar='E', help='test every N epochs')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--batch-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--batch-m', default=1, type=float,
                    metavar='N', help='m for negative sum')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='B', help='training batch size')
# parser.add_argument('--gpu', default='0,1,2,3', type=str,
#                     help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.batch_size = 128
# dataset = args.dataset 
dataset = "cache"
img_size = 64
pool_len = (16, 8)
if dataset == 'cifar':
    img_size = 32
    pool_len = 4
elif dataset == 'stl':
    img_size = 96
    pool_len = 7

log_dir = args.log_dir + dataset + '_log/'
test_epoch = args.test_epoch
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

suffix = dataset + '_batch_0nn_{}'.format(args.batch_size)
suffix = suffix + '_temp_{}_km_{}_alr'.format(args.batch_t, args.batch_m)

if len(args.resume) > 0:
    suffix = suffix + '_r'

# log the output
test_log_file = open(log_dir + suffix + '.txt', "w")
vis_log_dir = log_dir + suffix + '/'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Preparation
print('==> Preparing data..')
# np_dir = "/export/d2/zliudc/VirtualEnv/DeepCache/cache_log/"
# np_dir = "/home/wasmith/Music/pinTrace/cache_log"

"""
np_dir = "../cache_log/"
trainset = CacheDataset.CachePicDataset(np_dir)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size, shuffle=True,  # num_workers=4,
                                          drop_last=True)
testset = CacheDataset.CachePicDataset(np_dir, train=False)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100, shuffle=False, # num_workers=4
                                         )
"""

np_dir = os.path.abspath("../pin_dataset/pin_dataset_glow/")
trainset = CacheDataset.LargeCachePicDataset(np_dir, compiler="glow")
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
testset = CacheDataset.LargeCachePicDataset(np_dir, train=False, compiler="glow")
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100, shuffle=False, # num_workers=4
                                         )

"""
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if dataset == 'cifar':
    # cifar-10 dataset
    trainset = datasets.CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size, shuffle=True,  # num_workers=4,
                                              drop_last=True)

    testset = datasets.CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100, shuffle=False, # num_workers=4
                                             )
elif dataset == 'stl':
    # stl-10 dataset
    trainset = datasets.STL10Instance(root='./data', split='train+unlabeled', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    valset = datasets.STL10Instance(root='./data', split='train', download=True, transform=transform_test)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=100, shuffle=False, num_workers=4, drop_last=True)

    nvdata = valset.__len__()
    testset = datasets.STL10Instance(root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100, shuffle=False, num_workers=4)
"""

ndata = trainset.__len__()

print('==> Building model..')
net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)

# define leminiscate: inner product within each mini-batch (Ours)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# define loss function: inner product loss within each mini-batch
criterion = BatchCriterion(args.batch_m, args.batch_t, args.batch_size, device=device)

net.to(device)
criterion.to(device)

if args.test_only or len(args.resume) > 0:
    # Load checkpoint.
    model_path = args.model_dir + args.resume
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.test_only:
    if dataset == 'cifar':
        acc = kNN(epoch, net, trainloader, testloader, 200, args.batch_t, ndata, low_dim=args.low_dim)
    elif dataset == 'stl':
        acc = kNN(epoch, net, valloader, testloader, 200, args.batch_t, nvdata, low_dim=args.low_dim)
    sys.exit(0)

# define optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed at 120, 160 and 200"""
    lr = args.lr
    if epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    elif epoch >= 160 and epoch < 200:
        lr = args.lr * 0.05
    elif epoch >= 200:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    writer.add_scalar('lr', lr, epoch)


def adjust_learning_rate_quick(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed at 120, 160 and 200"""
    lr = args.lr
    if epoch >= 2 and epoch < 6:
        lr = args.lr * 0.1
    elif epoch >= 6 and epoch < 10:
        lr = args.lr * 0.05
    elif epoch >= 10:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    writer.add_scalar('lr', lr, epoch)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate_quick(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs1, inputs2, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)

        # print(targets)
        # input("continue?")
        inputs1, inputs2, indexes = inputs1.to(device), inputs2.to(device), indexes.to(device)

        inputs = torch.cat((inputs1, inputs2), 0)
        optimizer.zero_grad()

        features = net(inputs)
        loss = criterion(features, indexes)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
    # add log
    writer.add_scalar('loss', train_loss.avg, epoch)


def predict():
    # model_name = "resnet_embed_large_conv_3_10.net"
    # load model
    with open(model_name, 'rb') as f:
        checkpoint = torch.load(f)
    # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
    net.to("cpu")
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    # print(testset.id2name)
    # print(testset.next_ids)
    # print(testset.name_id)
    # print(testset.targets)
    labels = kNN_labels(net, trainset, testset, trainloader, testloader, 10, args.batch_t, ndata, low_dim=args.low_dim, device=device)
    with open("tmp_labels.json", 'w') as f:
        json.dump(labels, f, indent=2)


def single_embedding():
    print(len(trainset))
    # model_name = "resnet_embed_large_conv_3_10.net"
    # load model
    with open(model_name, 'rb') as f:
        checkpoint = torch.load(f)
    # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
    net.to("cpu")
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    net.to(device)
    # target1 = "resnet18-v2-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_compute_"
    # target1= "inception-v1-3+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_39_compute_"
    # target2 = "resnet18-v1-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_compute_"

    target1 = "resnet34-v1-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2_compute_"  # wrong
    # target1 = "resnet18-v2-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_1_compute_"
    target2 = "resnet18-v1-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2_compute_"
    embedding_list1 = []
    embedding_list2 = []
    inputs1 = []
    for idx in range(len(trainset)):
        if trainset.targets[idx*2] == trainset.name2id[target1]:
            inputs1.append(trainset.cache_pics[idx*2].astype(np.float32))
    inputs2 = []
    for idx in range(len(testset)):
        if testset.targets[idx] == testset.name2id[target2]:
            inputs2.append(testset.cache_pics[idx].astype(np.float32))
    inputs1 = torch.FloatTensor(inputs1)
    inputs2 = torch.FloatTensor(inputs2)
    # inputs1 = inputs1.cuda() if device == 'cuda' else inputs1
    # inputs2 = inputs2.cuda() if device == 'cuda' else inputs2
    print(inputs1.shape)
    print(inputs2.shape)

    output1 = net(inputs1)
    output2 = net(inputs2)
    print(output1.shape)
    print(output2.shape)

    # print(output1[0])
    # print(output2[0])
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos1 = nn.CosineSimilarity(dim=1, eps=1e-6)

    # cos_sim = cos1(output1, output2[1].repeat(55, 1))
    # print(cos_sim)
    
    # print(output2[1].repeat(1, 1).shape)
    tmp = torch.Tensor(output1.data.t())
    # print(tmp.shape)
    cos_sim = torch.mm(output2[0].repeat(1, 1), tmp)
    # print(cos_sim)
    # print(cos_sim.shape)
    print(cos_sim.topk(10, dim=1, largest=True, sorted=True))

    # Try with a single input
    # tmp = net(torch.FloatTensor([trainset.cache_pics[29346*2].astype(np.float32)]))
    # print(trainset.targets[29346*2])
    # print(trainset.id2name[trainset.targets[29346*2]])
    # print(tmp)
    # print("should be high", cos(tmp[0], output2[0]))


def generate_embedding_database(embedding_file="../pin_dataset/pin_dataset_glow/embeddings.json"):
    print(len(trainset))
    # model_name = "resnet_embed_large_conv_3_10.net"
    # load model
    with open(model_name, 'rb') as f:
        checkpoint = torch.load(f)
    # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
    net.to("cpu")
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    net.to(device)

    embedding_dict = {}
    for target_id in range(trainset.name_id):
        print(target_id, end=' ')
        if target_id % 10 == 0:
            print()
        # collect related traces
        inputs = []
        for idx in range(len(trainset.cache_pics)):
            if trainset.targets[idx] == target_id:
                inputs.append(trainset.cache_pics[idx].astype(np.float32))
        inputs = torch.FloatTensor(inputs)
        embeddings = net(inputs)
        embedding_dict[target_id] = embeddings.tolist()
    with open(embedding_file, 'w') as f:
        json.dump(embedding_dict, f)



def database_matching(embedding_file="../pin_dataset/pin_dataset_glow/embeddings.json", output_labels_path="distinct_labels_glow.json"):
    
    def trace_similarity(test_embeddings, target_embeddings):
        tmp = torch.Tensor(target_embeddings.data.t()).to(device)
        cos_sim = torch.mm(test_embeddings, tmp)
        # print(cos_sim.shape)  # debug
        # print(cos_sim)
        # cos_sim_mean = torch.mean(cos_sim, dim=1, keepdim=True)
        cos_sim_mean = torch.mean(cos_sim, dim=1)
        # print(cos_sim_mean)
        # print(cos_sim_mean.shape)  # debug
        sim = torch.mean(cos_sim_mean)
        # print(sim)
        
        return sim

    print(len(testset))
    # model_name = "resnet_embed_large_conv_3_10.net"
    # load model
    with open(model_name, 'rb') as f:
        checkpoint = torch.load(f)
    # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
    net.to("cpu")
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    net.to(device)

    # load pre-computed embeddings
    with open(embedding_file, 'r') as f:
        embedding_dict = json.load(f)

    output_list = []
    # for each input layer, we computed the average similarity scores and choose top-k candidates
    for test_id in range(testset.name_id):
        input_label = testset.id2name[test_id]
        print(testset.id2name[test_id])
        
        # collect related traces
        inputs = []
        for idx in range(len(testset.cache_pics)):
            if testset.targets[idx] == test_id:
                inputs.append(testset.cache_pics[idx].astype(np.float32))
        inputs = torch.FloatTensor(np.array(inputs))
        test_embeddings = net(inputs)

        # try to match detabase
        sim_values = []
        for target_id in range(len(embedding_dict)):
            # Note that json treats keys as strings
            target_embeddings = torch.FloatTensor(embedding_dict[str(target_id)])
            sim = trace_similarity(test_embeddings, target_embeddings)
            sim_values.append(sim)
        sim_values = torch.Tensor(sim_values)
        # print(sim_values)
        # print(sim_values.shape)
        yd, yi = sim_values.topk(100, dim=0, largest=True, sorted=True)  # TOP K
        # print(yd, yi)
        # input("debug")


        output_labels = []
        for label_id in yi:
            output_labels.append(trainset.id2name[label_id.item()])  # key should be ints
        output_list.append((input_label, output_labels))
    with open(output_labels_path, 'w') as f:
        json.dump(output_list, f, indent=2)

epoch = 10
model_name = "embedding_glow_{}.net".format(epoch)

if __name__ == '__main__':
    # print(trainset.name_id)
    # print(testset.name_id)
    # generate_embedding_database()
    database_matching()
    exit(0)
    
    # predict()
    # single_embedding()
    # exit(0)
    
    # torch.set_printoptions(edgeitems=10)
    # torch.set_printoptions(threshold=10000)
    # torch.set_printoptions(profile="full")

    for epoch in range(start_epoch, start_epoch + 11):

        # training
        train(epoch)

        # # save model
        # model_name = "resnet_embed_large{}.net".format(epoch)
        # net.to("cpu")
        # checkpoint = {'epochs': epoch,
        #               'state_dict': net.state_dict(), }
        # with open(model_name, 'wb') as f:
        #     torch.save(checkpoint, f)
        
        # exit(0)

        # # load model
        # with open(model_name, 'rb') as f:
        #     checkpoint = torch.load(f)
        # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
        # net.load_state_dict(checkpoint['state_dict'])
        # net.eval()

        # testing every test_epoch
        # test_epoch = 10
        if False and epoch % test_epoch == 0:
            net.eval()
            print('----------Evaluation---------')
            start = time.time()

            if dataset == 'cache':
                acc = kNN(epoch, net, trainloader, testloader, 10, args.batch_t, ndata, low_dim=args.low_dim, device=device)
            elif dataset == 'cifar':
                acc = kNN(epoch, net, trainloader, testloader, 200, args.batch_t, ndata, low_dim=args.low_dim)
            elif dataset == 'stl':
                acc = kNN(epoch, net, valloader, testloader, 200, args.batch_t, nvdata, low_dim=args.low_dim)

            print("Evaluation Time: '{}'s".format(time.time() - start))
            writer.add_scalar('nn_acc', acc, epoch)

            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir(args.model_dir):
                    os.mkdir(args.model_dir)
                torch.save(state, args.model_dir + suffix + '_best.t')
                best_acc = acc

            print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc))
            print('[Epoch]: {}'.format(epoch), file=test_log_file)
            print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc), file=test_log_file)
            test_log_file.flush()
    
    # save model
    # model_name = "resnet_embed_large_conv_3_{}.net".format(epoch)
    net.to("cpu")
    checkpoint = {'epochs': epoch,
                  'state_dict': net.state_dict(), }
    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)
