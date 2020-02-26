from utils.data_preprocess import LaneDataset, Totensor
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_data = LaneDataset('train.csv', transform=Totensor())
    train_loader = DataLoader(train_data, batch_size=16, shuffle=False, **kwargs)
    result = {i:0 for i in range(8)}
    trainprocess = tqdm(train_loader)
    for batch in trainprocess:
        mask = batch['mask'].numpy()
        for i in range(8):
            result[i] += np.sum(mask == i)
    for i in range(8):
        print('No {} has :  {} '.format(i, result[i]))

if __name__ == '__main__':
    main()