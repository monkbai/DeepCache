import json

import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
from model.conv_encoder import MSCRED
from utils.data import load_data_multi, load_data
import matplotlib.pyplot as plt
import numpy as np
import os


batch_size = 1

def train(dataLoader, model, optimizer, epochs, device):
    global batch_size

    model = model.to(device)
    print("------training on {}-------".format(device))
    for epoch in range(epochs):
        train_l_sum, n = 0.0, 0
        for x in tqdm(dataLoader):
            if x.shape[0] < batch_size:
                continue  # skip the last batch

            # print(x.shape)
            # input("debug")
            if batch_size > 1:
                shp = x.shape
                x_in = x.reshape([batch_size*5, shp[2], shp[3], shp[4]])
            else:
                x_in = x.squeeze()
            # print(x_in.shape)
            x_in = x_in.to(device)
            out = model(x_in)
            # print(out.shape)
            # input("debug")
            
            if batch_size == 1:
                target = x[-1].unsqueeze(0).to(device)
                # target = x
            else:
                target = x[:,-1].to(device)

            l = torch.mean((out - target) ** 2)
            train_l_sum += l
            # print(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            # print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))

        print("[Epoch %d/%d] [loss: %f]" % (epoch + 1, epochs, train_l_sum / n))


def test(dataLoader, model, device, max_len=-1):
    global batch_size

    print("------Testing-------")
    index = 30
    loss_list = []
    reconstructed_data_path = "./data/matrix_data/reconstructed_data/"
    
    if max_len == -1:
        max_len = len(dataLoader.dataset)
    else:
        max_len = max(max_len, len(dataLoader.dataset))

    with torch.no_grad():
        for x in tqdm(dataLoader):  # for idx in range(len(dataLoader.dataset)):  # for x in dataLoader:
            # x = dataLoader.dataset[idx]
            if x.shape[0] < batch_size:
                continue  # skip the last batch
            
            if batch_size > 1:
                shp = x.shape
                x_in = x.reshape([batch_size*5, shp[2], shp[3], shp[4]])
            else:
                x_in = x.squeeze()
            x_in = x_in.to(device)
            reconstructed_matrix = model(x_in)

            if batch_size == 1:
                target = x[-1].unsqueeze(0).to(device)
                # target = x
            else:
                target = x[:,-1].to(device)
            # target = x

            if batch_size == 1:
                loss = torch.mean((reconstructed_matrix - target) ** 2)
                loss_list.append([index, loss.item()])
                indx += 1
            else:
                tmp = (reconstructed_matrix - target)
                for idx in range(batch_size):
                    loss = torch.mean(tmp[idx] ** 2)
                    loss_list.append([index, loss.item()])
                    index += 1 
            # path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
            # np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            # # l = criterion(reconstructed_matrix, x[-1].unsqueeze(0)).mean()
            # # loss_list.append(l)
            # # print("[test_index %d] [loss: %f]" % (index, l.item()))
            # index += 1
    with open("./loss_list.json", 'w') as f:
        json.dump(loss_list, f, indent=2)


def batch_size_1():
    global batch_size

    trace_file_prefix = "/home/wasmith/Music/pinTrace/cache_log/0x402240-0x4029ed"
    trace_file = "/export/d2/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_glow/resnet18_v1_7.out/0036.libjit_convDKKC8_f_5-0x4090f0-0x409b08.log"
    trace_file_prefix = trace_file[:-4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    # # data_loader = load_data(trace_file_prefix, batch_size=32, size=5000)
    # batch_size = 32
    # data_loader = load_data_multi(trace_file_prefix, batch_size=batch_size, size=5000)
    # mscred = MSCRED(3, 256, batch_size)

    # # train stage
    # # mscred.load_state_dict(torch.load("./checkpoints/model1.pth"))
    # optimizer = torch.optim.Adam(mscred.parameters(), lr=0.0002)
    # train(data_loader, mscred, optimizer, epochs=1, device=device)
    # print("Saving model....")
    # torch.save(mscred.state_dict(), "./checkpoints/model_test.pth")

    # test stage
    batch_size = 32
    mscred = MSCRED(3, 256, batch_size)
    mscred.load_state_dict(torch.load("./checkpoints/model_test.pth"))
    mscred.to(device)
    # data_loader = load_data(trace_file_prefix, batch_size=1, size=5000)
    data_loader = load_data_multi(trace_file_prefix, batch_size=batch_size, size=5000)
    test(data_loader, mscred, device, max_len=5000)

    exit(0)


def seg_ids(loss_file: str):
    seg_ids = []
    with open(loss_file, 'r') as f:
        loss_list = json.load(f)
    loss_values = [x[1] for x in loss_list]
    avg_loss = sum(loss_values) / len(loss_values)
    print("avg_loss", avg_loss)
    for idx, loss in loss_list:
        if loss > avg_loss * 2:
            seg_ids.append(idx)
    print(seg_ids)


if __name__ == '__main__':
    seg_ids("loss_list.json")
    exit(0)
    batch_size_1()

    trace_file = "/home/wasmith/Music/pinTrace/cache_log/0x402240-0x4029ed.log"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    # data_loader = load_data(trace_file)
    data_loader = load_data_multi(trace_file)
    mscred = MSCRED(3, 256)

    # train stage
    # mscred.load_state_dict(torch.load("./checkpoints/model1.pth"))
    optimizer = torch.optim.Adam(mscred.parameters(), lr=0.0002)
    train(data_loader, mscred, optimizer, epochs=10, device=device)
    print("Saving model....")
    torch.save(mscred.state_dict(), "./checkpoints/model_test.pth")

    # test stage
    mscred.load_state_dict(torch.load("./checkpoints/model_test.pth"))
    mscred.to(device)
    test(data_loader, mscred)
