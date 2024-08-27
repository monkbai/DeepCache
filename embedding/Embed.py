import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import gc
import os
import argparse
import time
import json
import numpy as np
import embedding.models as models

from embedding.BatchAverage import BatchCriterion
from embedding.utils import *

import embedding.CacheDataset as CacheDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
pool_len = (16, 8)
low_dim = 128
batch_size = 64  # batch_size = 128
batch_m = 1
batch_t = 0.1
ndata = 0
lr = 0.03

epoch = 10
model_name = "embedding_glow_{}.net".format(epoch)

mat_mode = False
# mat_mode = True
if mat_mode:
    pool_len = (8, 8)

def set_model_name(compiler='tvm', epoch_num=10, prefix='embedding'):
    global epoch, model_name
    epoch = epoch_num
    model_name = "{}_{}_{}.net".format(prefix, compiler, epoch)


def dataset(np_dir="./pin_dataset/pin_dataset/", compiler='tvm', paral_ver=False):
    """ np_dir: where numpy arrarys are stored """
    global ndata, mat_mode

    np_dir = os.path.abspath(np_dir)
    if not mat_mode:
        trainset = CacheDataset.LargeCachePicDataset(np_dir, compiler=compiler, paral_ver=paral_ver)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size, shuffle=True, num_workers=4,
                                                drop_last=True)
        testset = CacheDataset.LargeCachePicDataset(np_dir, train=False, compiler=compiler)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size, shuffle=False, # num_workers=4
                                                )
    else:
        trainset = CacheDataset.LargeCacheMatDataset(np_dir, compiler=compiler)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size, shuffle=True, num_workers=4,
                                                drop_last=True)
        testset = CacheDataset.LargeCacheMatDataset(np_dir, train=False, compiler=compiler)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=100, shuffle=False, # num_workers=4
                                                )
    
    ndata = trainset.__len__()
    print("len(trainset):", ndata)
    print("len(trainset.name2id):", len(trainset.name2id))
    print("trainset.name_id:", trainset.name_id)
    # print(trainset.name2id)
    ndata = testset.__len__()
    print("len(testset):", ndata)
    print("len(testset.name2id):", len(testset.name2id))
    print("testset.name_id:", testset.name_id)
    return trainset, trainloader, testset, testloader


def database(np_dir="./pin_dataset/pin_dataset/", compiler='tvm'):
    """ np_dir: where numpy arrarys are stored """
    global ndata, mat_mode

    np_dir = os.path.abspath(np_dir)
    if not mat_mode:
        trainset = CacheDataset.LargeCachePicDataset(np_dir, compiler=compiler, as_database=True)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size, shuffle=True, num_workers=4,
                                                drop_last=True)
        testset = CacheDataset.LargeCachePicDataset(np_dir, train=False, compiler=compiler, as_database=True)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size, shuffle=False, # num_workers=4
                                                )
    else:
        trainset = CacheDataset.LargeCacheMatDataset(np_dir, compiler=compiler)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size, shuffle=True, num_workers=4,
                                                drop_last=True)
        testset = CacheDataset.LargeCacheMatDataset(np_dir, train=False, compiler=compiler)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=100, shuffle=False, # num_workers=4
                                                )
            
    ndata = trainset.__len__()
    print("len(trainset):", ndata)
    print("len(trainset.name2id):", len(trainset.name2id))
    print("trainset.name_id:", trainset.name_id)
    # print(trainset.name2id)
    ndata = testset.__len__()
    print("len(testset):", ndata)
    print("len(testset.name2id):", len(testset.name2id))
    print("testset.name_id:", testset.name_id)
    return trainset, trainloader, testset, testloader


def build_model(pool_len=pool_len, low_dim=low_dim, lr=lr):
    net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=low_dim)
    
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #     cudnn.benchmark = True
    net.to(device)

    # define loss function: inner product loss within each mini-batch
    criterion = BatchCriterion(batch_m, batch_t, batch_size, device=device)
    criterion.to(device)

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return net, criterion, optimizer


def adjust_learning_rate(optimizer, epoch):
    global lr
    
    # if epoch >= 2 and epoch < 4:
    #     lr = lr # * 0.1
    # elif epoch >= 4 and epoch < 6:
    #     lr = lr # * 0.05
    # elif epoch >= 6:
    #     lr = lr # * 0.01
    lr_tmp = lr
    if epoch >= 40:
        lr_tmp = lr * 0.5
    if epoch >= 80:
        lr_tmp = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_tmp


def train_step(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()
    net.to(device)

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

        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
    
    return net, criterion, optimizer, train_loss.avg
    

def train(net, criterion, optimizer, trainloader, testloader, start_epoch=0):
    global epoch
    print("epoch num:", epoch)

    reuse_model = True
    if os.path.exists(model_name) and reuse_model:  # TODO: skip
        # load model
        with open(model_name, 'rb') as f:
            checkpoint = torch.load(f)
        # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
        net.to("cpu")
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        net.to(device) 
        # net.to("cuda:1")
        # epoch = 10  # further finetune  
        return model_name  
        start_epoch = checkpoint['epochs']
    # else:
    best_loss = 10
    net.to(device)
    # for epoch_num in range(start_epoch, start_epoch + epoch):
    #     print(epoch_num, end=' ')
    print("")
    for epoch_num in range(start_epoch, start_epoch + epoch):
        net, criterion, optimizer, train_loss = train_step(net, criterion, optimizer, epoch_num, trainloader)
        
        print(train_loss)
        if train_loss < best_loss:
            print('Saving...(loss: {})'.format(train_loss))
            # save model
            # model_name = "resnet_embed_large_conv_3_{}.net".format(epoch)
            net.to("cpu")
            checkpoint = {'epochs': epoch_num,
                            'loss': train_loss,
                        'state_dict': net.state_dict(), }
            with open(model_name, 'wb') as f:
                torch.save(checkpoint, f)
            best_loss = train_loss
        
    print("Best Loss:", best_loss)
    return model_name


def generate_embedding_database(net, trainset, embedding_file="../pin_dataset/pin_dataset_glow/embeddings.json"):
    global model_name 

    print("dataset size:", len(trainset))
    torch.cuda.empty_cache()
    
    # load model
    with open(model_name, 'rb') as f:
        checkpoint = torch.load(f)
    # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
    net.to("cpu")
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    net.to(device) 
    # net.to("cuda:1")

    embedding_dict = {}
    for target_id in range(trainset.name_id):
        # print(target_id, end=' ')  # debug
        # if target_id % 20 == 0:
        #     print()
        
        # collect related traces
        inputs = []
        for idx in range(len(trainset.cache_pics)):
            if trainset.targets[idx] == target_id:
                inputs.append(trainset.cache_pics[idx].astype(np.float32))
        inputs = torch.FloatTensor(np.array(inputs)).to("cpu")
        if inputs.shape[0] > 64:
            inputs_list = torch.split(inputs, 64)
            del inputs
            # torch.cuda.empty_cache()
            embeddings = []
            for idx in range(len(inputs_list)):
                cur_inputs = inputs_list[idx].to(device)
                # print(idx, cur_inputs.shape)
                cur_embeddings = net(cur_inputs)
                _embeddings = cur_embeddings.to("cpu")
                embeddings.append(_embeddings)

            del inputs_list, cur_inputs, cur_embeddings
            gc.collect()
            torch.cuda.empty_cache()

            all_embeddings = torch.stack(embeddings[:-1]).to("cpu")
            all_embeddings = torch.flatten(all_embeddings, 0, 1)
            # print("embeddings shape", all_embeddings.shape)
            all_embeddings = torch.cat((all_embeddings, embeddings[-1])).to("cpu")
            
            # print("embeddings shape", all_embeddings.shape)
            # input("debug")
            embedding_dict[target_id] = all_embeddings.tolist()
            del embeddings, all_embeddings
        else:
            embeddings = net(inputs)
            embedding_dict[target_id] = embeddings.tolist()
    with open(embedding_file, 'w') as f:
        json.dump(embedding_dict, f)

    return embedding_file



def database_matching(net, trainset, testset, 
                      embedding_file="../pin_dataset/pin_dataset_glow/embeddings.json", 
                      output_labels_path="distinct_labels_glow.json", 
                      LLC=False, topk=100, dis_thre_min=0.55, dis_thre_max=None):
    """ topk used in this function is not topk for the final candidate """

    def trace_similarity_LLC(test_embeddings, target_embeddings, 
                         input_label='', target_label=''):
        tmp = torch.Tensor(target_embeddings.data.t()).to(device)
        cos_sim = torch.mm(test_embeddings, tmp)
        
        # cos_sim_mean = torch.mean(cos_sim, dim=1)
        # cos_sim_mean_pos = cos_sim_mean[cos_sim_mean > 0.4]
        sim = -1.0
        
        # Plan 4:
        cos_sim_flat = torch.flatten(cos_sim)
        cos_sim_flat = cos_sim_flat[cos_sim_flat > dis_thre_min] # 0.55
        if dis_thre_max:
            if True in (cos_sim_flat > dis_thre_max):
                # print(cos_sim_flat > dis_thre_max)
                # print('before', cos_sim_flat)
                # print(cos_sim_flat[cos_sim_flat > dis_thre_max])
                cos_sim_flat = cos_sim_flat[cos_sim_flat <= dis_thre_max]
                # print('after', cos_sim_flat)
                # input("continue?")
        sim = len(cos_sim_flat) / (11.0*11.0) 

        # # Plan 3:
        # sim = len(cos_sim_mean_pos) / 110.0

        # # Plan 2:
        # if len(cos_sim_mean_pos) <= 0:
        #     sim = -1.0 
        # else:
        #     sim = torch.mean(cos_sim_mean_pos)
        
        # # Plan 1:
        # sim = torch.mean(cos_sim_mean)
        
        # if sim > 0.15:
        #     print(input_label + "-->" + target_label)
        #     # print(test_embeddings.shape, target_embeddings.shape)
        #     # print("cos_sim")
        #     # print(cos_sim.shape)
        #     # print(cos_sim)  # debug
        #     print(torch.mean(cos_sim[0]))  # debug
        #     print("cos_sim_mean")
        #     print(cos_sim_mean)  # debug
        #     print("cos_sim_mean_pos")
        #     print(cos_sim_mean_pos.shape)
        #     print(cos_sim_mean_pos)  # debug
        #     print(sim)
        #     input("continue?")

        return sim
    
    def trace_similarity_internal(test_embeddings, target_embeddings, 
                         input_label='', target_label=''):
        tmp = torch.Tensor(target_embeddings.data.t()).to(device)
        cos_sim = torch.mm(test_embeddings, tmp)
        
        cos_sim_mean = torch.mean(cos_sim, dim=1)
        # cos_sim_mean_pos = cos_sim_mean[cos_sim_mean > 0.4]
        
        # Plan 1:
        sim = torch.mean(cos_sim_mean)

        return sim
    
    def trace_similarity(test_embeddings, target_embeddings, 
                         input_label='', target_label=''):
        if LLC:
            return trace_similarity_LLC(test_embeddings, target_embeddings, 
                                        input_label, target_label)
        else:
            return trace_similarity_internal(test_embeddings, target_embeddings, 
                                        input_label, target_label)

    print("testset size", len(testset))
    
    # load model
    with open(model_name, 'rb') as f:
        checkpoint = torch.load(f)
    # net = models.__dict__['ResNet18'](pool_len=pool_len, low_dim=args.low_dim)
    net.to("cpu")
    # net.to(device) # no enough memory
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    # net.to(device)

    # load pre-computed embeddings
    with open(embedding_file, 'r') as f:
        embedding_dict = json.load(f)

    output_list = []
    # for each input layer, we computed the average similarity scores and choose top-k candidates
    for test_id in range(testset.name_id):
        input_label = testset.id2name[test_id]
        print("Getting similarity scores for", testset.id2name[test_id])
        
        # # debug only
        # if 'resnet18-v1-7-loop+tvmgen_default_fused_nn_contrib_dense_pack_add_' not in input_label:
        #     continue

        # collect related traces
        inputs = []
        for idx in range(len(testset.cache_pics)):
            if testset.targets[idx] == test_id:
                inputs.append(testset.cache_pics[idx].astype(np.float32))
        inputs = torch.FloatTensor(np.array(inputs)).to("cpu")  
        # inputs = torch.FloatTensor(np.array(inputs)).to(device)
        # print(inputs.shape)
        test_embeddings = net(inputs)
        test_embeddings = test_embeddings.to(device)

        # try to match detabase
        sim_values = []
        for target_id in range(len(embedding_dict)):
            # Note that json treats keys as strings
            target_embeddings = torch.FloatTensor(embedding_dict[str(target_id)])
            target_label = trainset.id2name[target_id]
            sim = trace_similarity(test_embeddings, target_embeddings, input_label, target_label)
            sim_values.append(sim)
        sim_values = torch.Tensor(sim_values)
        # print(sim_values)
        # print(sim_values.shape)
        yd, yi = sim_values.topk(topk, dim=0, largest=True, sorted=True)  # TOP K
        # print(yd, yi)
        # input("debug")


        output_labels = []
        for idx in range(len(yi)):
            label_id = yi[idx]
            sim_val = yd[idx]
            output_labels.append((trainset.id2name[label_id.item()], sim_val.item()))  # key should be ints
        output_list.append((input_label, output_labels))
    with open(output_labels_path, 'w') as f:
        json.dump(output_list, f, indent=2)

    return output_labels_path
