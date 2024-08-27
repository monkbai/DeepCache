import time
import copy
import json

import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn
from sklearn.cluster import KMeans


def kNN_labels(net, traindataset, testdataset, trainloader, testloader, K, sigma, ndata, low_dim = 128, device='cuda'):
    net.to(device)
    net.eval()
    # net_time = AverageMeter()
    # cls_time = AverageMeter()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

   
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        try:
            # trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
            
            # TODO: Warning
            tmp = copy.deepcopy(trainloader.dataset.targets)
            new_targets = []
            for i in range(0, len(tmp), 2):
                new_targets.append(tmp[i])

            if device == "cuda":
                trainLabels = torch.LongTensor(new_targets).cuda()
            else:
                trainLabels = torch.LongTensor(new_targets)
        except:
            trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    
    trainFeatures = np.zeros((low_dim, ndata))    
    C = trainLabels.max() + 1
    # C = np.int(C)
    with torch.no_grad():
        # transform_bak = trainloader.dataset.transform
        # trainloader.dataset.transform = testloader.dataset.transform
        batch_size = 100
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=batch_size, shuffle=False)  # , num_workers=4)
        for batch_idx, (inputs, _, targets, indexes) in enumerate(temploader):
            # targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            #
            tmp = features.cpu().data.t()
            if batch_idx*batch_size+batch_size >= ndata:
                trainFeatures[:, batch_idx * batch_size:] = tmp
            else:
                trainFeatures[:, batch_idx*batch_size:batch_idx*batch_size+batch_size] = tmp
    # trainloader.dataset.transform = transform_bak
    # 

    trainFeatures = torch.Tensor(trainFeatures)
    if device == "cuda":
        trainFeatures = trainFeatures.cuda()
    # print(trainFeatures.shape)
    top1 = 0.
    top5 = 0.
    output_list = []
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C)
        retrieval_one_hot = retrieval_one_hot.cuda() if device == "cude" else retrieval_one_hot
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda() if device == 'cuda' else targets
            batchSize = inputs.size(0)  
            features = net(inputs)
            total += targets.size(0)

            dist = torch.mm(features, trainFeatures)
            # print("dist shape", dist.shape)
            # dist = torch.gt(dist, 0.9)
            # print("dist shape", dist.shape)
            # print(dist)
            # input("debug")
            # tmp = torch.count_nonzero(dist, dim=1)
            # print("tmp shape", tmp.shape)
            # print(tmp)
            # input("debug")  # roughly 55 candidates ( > 0.8) 
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            # print(yi)
            # input("debug")
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            # print(trainLabels.shape)
            # print(trainLabels.view(1,-1).expand(batchSize, -1).shape)
            # input("debug")

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot = retrieval_one_hot.to(device)
            tmp = retrieval.view(-1, 1).to(device)
            retrieval_one_hot.scatter_(1, tmp, 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # print(predictions.shape)
            # print(predictions)
            # print(targets.shape)
            # print(targets)
            # input("continue?")
            
            # Find which predictions match the target
            # correct = predictions.eq(targets.data.view(-1,1))

            for idx in range(len(retrieval)):
                input_id = targets[idx].item()
                input_label = testdataset.id2name[input_id]
                out_labels = []
                for label_idx in range(len(retrieval[idx])):
                    out_labels.append(traindataset.id2name[retrieval[idx][label_idx].item()])
                output_list.append((input_label, out_labels))

            # print(output_list)
            # input("continue?")

            # top1 = top1 + correct.narrow(1,0,1).sum().item()
            # top5 = top5 + correct.narrow(1,0,5).sum().item()

    print(len(output_list))
    return output_list 


def kNN(epoch, net, trainloader, testloader, K, sigma, ndata, low_dim = 128, device='cuda'):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

   
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        try:
            # trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
            if device == "cuda":
                trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
            else:
                trainLabels = torch.LongTensor(trainloader.dataset.targets)
        except:
            trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    trainFeatures = np.zeros((low_dim, ndata))    
    C = trainLabels.max() + 1
    # C = np.int(C)
    with torch.no_grad():
        # transform_bak = trainloader.dataset.transform
        # trainloader.dataset.transform = testloader.dataset.transform
        batch_size = 100
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=batch_size, shuffle=False)  # , num_workers=4)
        for batch_idx, (inputs, _, targets, indexes) in enumerate(temploader):
            # targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            #
            tmp = features.cpu().data.t()
            if batch_idx*batch_size+batch_size >= ndata:
                trainFeatures[:, batch_idx * batch_size:] = tmp
            else:
                trainFeatures[:, batch_idx*batch_size:batch_idx*batch_size+batch_size] = tmp
    # trainloader.dataset.transform = transform_bak
    # 

    trainFeatures = torch.Tensor(trainFeatures)
    if device == "cuda":
        trainFeatures = trainFeatures.cuda()
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C)
        retrieval_one_hot = retrieval_one_hot.cuda() if device == "cude" else retrieval_one_hot
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda() if device == 'cuda' else targets
            batchSize = inputs.size(0)  
            features = net(inputs)
            total += targets.size(0)

            
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot = retrieval_one_hot.to(device)
            tmp = retrieval.view(-1, 1).to(device)
            retrieval_one_hot.scatter_(1, tmp, 1)
                
            yd_transform = yd.clone().div_(sigma).exp_()
            mul_tmp = torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1))
            probs = torch.sum(mul_tmp, 1)
            # print(probs)
            _, predictions = probs.sort(1, True)

            if epoch > -1:
                # print(features)
                # print(dist)  # 100, N ?
                print(yd)
                print(retrieval)  # retrieved labels
                print(retrieval.shape)  # 100, 10
                # print(retrieval_one_hot)
                print(retrieval_one_hot.shape)  # 1000, 15
                # print(yd_transform)
                print(yd_transform.shape)  # 100, 10
                # print(retrieval_one_hot.view(batchSize, -1 , C))
                # print(yd_transform.view(batchSize, -1, 1))
                print(retrieval_one_hot.view(batchSize, -1 , C).shape)  # 100, 10, 15
                print(yd_transform.view(batchSize, -1, 1).shape)  # 100, 10, 1
                # print(mul_tmp)
                print(mul_tmp.shape)
                # print(probs)
                print(probs.shape)  # 100, 15
                print(predictions[0])
                print(predictions.shape)
                input("debug")

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            # print(predictions)
            # print(targets)
            # input("continue?")
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()
            # top1 = top1 + correct.narrow(1,1,1).sum().item()
            # top5 = top5 + correct.narrow(1,1,5).sum().item()


            print('Test [{}/{}]\t'
              'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
              'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
              'Top1: {:.2f}  Top5: {:.2f}'.format(
              total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)

    return top1*100./total 
    
def eval_nmi_recall(epoch, net, lemniscate, testloader, feature_dim = 128):    
    net.eval()
    net_time = AverageMeter()
    val_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()
    
    ptr =0
    nmi = 0.
    recal = 0.
    end = time.time()
    test_features = np.zeros((testsize,feature_dim))
    test_labels   = np.zeros(testsize)
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            # end = time.time()
            batchSize = inputs.size(0)
            real_size = min(batchSize, testsize-ptr)
            targets = np.asarray(targets)
            batch_features = net(inputs)
            test_features[ptr:ptr+real_size,:] = batch_features
            test_labels[ptr:ptr+real_size]  = targets
            ptr += real_size
    net_time.update(time.time() - end)
    print('Extracting Time:\t'
                  'Net Time {net_time.val:.3f}s \t'
                  .format(net_time=net_time))
    # 
    # print('Evaluating.....')
    end = time.time()
    recal = eval_recall(test_features,test_labels)
    nmi = eval_nmi(test_features, test_labels)
    val_time.update(time.time() - end)
    print('Evaluating Time:\t'
                  'Eval Time {val_time.val:.3f}s \t'
                  .format(val_time=val_time))
    return recal, nmi 
    
def eval_recall(embedding, label):
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0
    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        pred = np.argmin(dis)
        if label[i]==label[pred]:
            right_num = right_num+1
    recall = float(right_num)/float(embedding.shape[0])
    return recall

def eval_nmi(embedding, label,  normed_flag = False, fast_kmeans = False):
    unique_id = np.unique(label)
    num_category = len(unique_id)
    if normed_flag:
        for i in range(embedding.shape[0]):
            embedding[i,:] = embedding[i,:]/np.sqrt(np.sum(embedding[i,:] ** 2)+1e-4)
    if fast_kmeans:
        kmeans = KMeans(n_clusters=num_category, n_init = 1, n_jobs=8)
    else:
        kmeans = KMeans(n_clusters=num_category,n_jobs=8)
    kmeans.fit(embedding)
    y_kmeans_pred = kmeans.predict(embedding)
    nmi = normalized_mutual_info_score(label, y_kmeans_pred)
    return nmi

def eval_recall_K(embedding, label, K_list =None):
    if K_list is None:
        K_list = [1, 2, 4, 8]
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0

    recall_list = np.zeros(len(K_list))

    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        index = np.argsort(dis)
        list_index = 0
        for k in range(np.max(K_list)):
            if label[i]==label[index[k]]:
                recall_list[list_index] = recall_list[list_index]+1
                break
            if k>=K_list[list_index]-1:
                list_index = list_index + 1
    recall_list = recall_list/float(embedding.shape[0])
    for i in range(recall_list.shape[0]):
        if i == 0:
            continue
        recall_list[i] = recall_list[i]+recall_list[i-1]
    return recall_list
    
class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()      


def check_labels(predict_labels, attr_labels):
    with open(predict_labels, 'r') as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, 'r') as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)
    
    correct_count = 0
    all_count = 0
    for name, output_list in pre_labels:
        name = name.replace("_compute_", "")
        target = attr_labels[name]
        predict = attr_labels[output_list[0].replace("_compute_", "")]
        if target == predict:
            correct_count += 1
        else:
            print(name, output_list)
            input("debug")
        all_count += 1
    print("{}/{}".format(correct_count, all_count))


def check_vote_labels(predict_labels, attr_labels):
    from collections import Counter

    with open(predict_labels, 'r') as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, 'r') as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)
    
    correct_count = 0
    all_count = 0
    name2vote = {}
    for name, output_list in pre_labels:
        name = name.replace("_compute_", "")
        if name not in name2vote:
            name2vote[name] = []

        name2vote[name].append(output_list[0].replace("_compute_", ""))
    for name, output_list in name2vote.items():   
        target = attr_labels[name]
        counter = Counter(output_list)
        pre_name = counter.most_common(1)[0][0]
        predict = attr_labels[pre_name]
        if target == predict:
            correct_count += 1
        else:
            # print(name)
            print(name, target)
            print(counter.most_common(1), predict)
            # input("debug")
        all_count += 1
    print("{}/{}".format(correct_count, all_count))


def check_topk_labels(predict_labels, attr_labels, topk=1):
    with open(predict_labels, 'r') as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, 'r') as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)
    
    correct_count = 0
    all_count = 0
    for name, output_list in pre_labels:
        # if "vgg16-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2" not in name:
        #     continue

        name = name.replace("_compute_", "")
        target = attr_labels[name]
        # print("target", target)
        output_list = output_list[:topk]

        exist = False
        for pre_name in output_list:
            if isinstance(pre_name, list):
                pre_name = pre_name[0]
            pre_name = pre_name.replace("_compute_", "")
            predict = attr_labels[pre_name]
            # print("predict", predict)
            if target == predict:
                exist = True
                break
        if exist:
            correct_count += 1
        else:
            pass
            # print(name, output_list)
            # input("debug")
        all_count += 1
    print("{}/{}".format(correct_count, all_count))


def check_topk_labels_glow(predict_labels, attr_labels, topk=1):
    with open(predict_labels, 'r') as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, 'r') as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)
    
    correct_count = 0
    all_count = 0
    for name, output_list in pre_labels:
        # if "vgg16-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2" not in name:
        #     continue

        name = name.replace("_compute_", "")
        target = attr_labels[name]
        # print("target", target)
        output_list = output_list[:topk]

        exist = False
        for pre_name in output_list:
            pre_name = pre_name.replace("_compute_", "")
            predict = attr_labels[pre_name]
            # print("predict", predict)
            if target == predict:
                exist = True
                break
        if exist:
            correct_count += 1
        else:
            # print(name, output_list)
            # input("debug")
            pass
        all_count += 1
    print("{}/{}".format(correct_count, all_count))


if __name__ == '__main__':
    # check_vote_labels("./tmp_labels.json", "/export/d2/zliudc/VirtualEnv/ONNX_Zoo/TVM-0.12/labels.json")
    # check_topk_labels("./distinct_labels.json", "/export/d2/zliudc/VirtualEnv/ONNX_Zoo/TVM-0.12/labels.json", topk=8)

    # check_topk_labels_glow("./distinct_labels_glow.json", "/export/d2/zliudc/VirtualEnv/ONNX_Zoo/Glow-2023/labels_glow.json", topk=100)
    
    check_topk_labels("./embedding/distinct_labels_tvm_oram.json", 
                      # "/export/d2/zliudc/VirtualEnv/ONNX_Zoo/TVM-0.12/labels_new.json",
                      "./attr_labels_tvm.json", 
                      topk=30)
