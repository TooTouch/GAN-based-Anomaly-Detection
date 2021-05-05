import os
import pickle
import torch
import numpy as np
from dataset import BuildDataset
from sklearn.preprocessing import MinMaxScaler

def get_dataloader(
    data_root, data_name, batch_size, scale=None, window_size=60, slide_size=30, time_phase=True
):
    '''
    Return data loader

    Parameters:
        data_root (str): path of the data
        data_name (str): name of the data
        batch_size (int): batch size
        scale (str): scaling mathod name, ex) minmax
        window_size (int): window size for time series condition
        slide_size (int): moving window size
        time_phase (bool): i.i.d. if False

    Returns:
        train/dev/tst dataloader (torch.utils.data.DataLoader):
            output shape - time series: (batch size, window size, 31)
                           i.i.d: (batch size, 31)
    '''
    with open(os.path.join(data_root, data_name), 'rb') as f:
        df = pickle.load(f)

    if time_phase:
        # time phase condition
        print("time series condition")
        trn = df[df['time_phase'] == 'trn'].iloc[:,:-3].values
        dev = df[df['time_phase'] == 'val'].iloc[:,:-3].values
        tst = df.iloc[:,:-3].values
    else:
        # i.i.d. condition
        print("i.i.d. condition")
        trn = df[df['iid_phase'] == 'trn'].iloc[:,:-3].values
        dev = df[df['iid_phase'] == 'val'].iloc[:,:-3].values
        tst = df.iloc[:,:-3].values

    if scale=='minmax':
        scaler = MinMaxScaler()
        scaler.fit(trn)
        trn = scaler.transform(trn)
        dev = scaler.transform(dev)
        tst = scaler.transform(tst)
    elif scale=='minmax_square':
        scaler = MinMaxScaler()
        scaler.fit(trn)
        trn = scaler.transform(trn) ** 2
        dev = scaler.transform(dev) ** 2
        tst = scaler.transform(tst) ** 2
    elif scale=='minmax_m1p1':
        trn = 2 * (trn / trn.max(axis=0)) - 1
        dev = 2 * (dev / dev.max(axis=0)) - 1
        tst = 2 * (tst / tst.max(axis=0)) - 1

    trn_dataset = BuildDataset(trn, window_size, slide_size, time_phase)
    dev_dataset = BuildDataset(dev, window_size, slide_size, time_phase)
    tst_dataset = BuildDataset(tst, window_size, 1, time_phase, test=True)
    
    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    return trn_dataloader, dev_dataloader, tst_dataloader