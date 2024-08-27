import torch
import numpy as np
import os

splits = ["train", "test"]
train_data_path = "./data/matrix_data/train_data/"
test_data_path = "./data/matrix_data/test_data/"
shuffle = {'train': True, 'test': False}

def load_data_old():
    dataset = {}
    train_file_list = os.listdir(train_data_path)
    test_file_list = os.listdir(test_data_path)
    train_file_list.sort(key = lambda x:int(x[11:-4]))
    test_file_list.sort(key = lambda x:int(x[10:-4]))
    train_data, test_data = [],[]
    for obj in train_file_list:   
        train_file_path = train_data_path + obj
        train_matrix = np.load(train_file_path)
        #train_matrix = np.transpose(train_matrix, (0, 2, 3, 1))
        train_data.append(train_matrix)

    for obj in test_file_list:
        test_file_path = test_data_path + obj
        test_matrix = np.load(test_file_path)
        #test_matrix = np.transpose(test_matrix, (0, 2, 3, 1))
        test_data.append(test_matrix)

    dataset["train"] = torch.from_numpy(np.array(train_data)).float()
    dataset["test"] = torch.from_numpy(np.array(test_data)).float()

    dataloader = {x: torch.utils.data.DataLoader(
                                dataset=dataset[x], batch_size=1, shuffle=shuffle[x]) 
                                for x in splits}
    return dataloader


def load_data(file_prefix: str, batch_size=1, size=-1):
    prefix = file_prefix
    data_vec = []
    for win in [10, 20, 30]:
        tmp_data = np.load(prefix + '_win_' + str(win) + '.npy')
        data_vec.append(tmp_data[30:])
    data_vec = np.array(data_vec)
    data_vec = np.transpose(data_vec, (1, 0, 2, 3))

    if size != -1:
        data_vec = data_vec[:size]

    data_tensor = torch.from_numpy(data_vec).float()
    data_loader = torch.utils.data.DataLoader(dataset=data_tensor, batch_size=batch_size, shuffle=False)
    return data_loader


def load_data_multi(file_prefix: str, batch_size=1, size=-1):
    prefix = file_prefix  # "/home/wasmith/Music/pinTrace/cache_log/0x402240-0x4029ed"
    data_vec = []
    data_vec = np.load(prefix + '_multi5.npy')

    if size != -1:
        data_vec = data_vec[:size]

    data_vec = np.array(data_vec)
    data_tensor = torch.from_numpy(data_vec).float()
    data_loader = torch.utils.data.DataLoader(dataset=data_tensor, batch_size=batch_size, shuffle=False)
    # print(len(data_vec))
    return data_loader
