import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader
from io_utils import SEEDDataset_raw
from load_data import load_seed_raw_trainVal
from simCLR import SimCLR
from train_utils import train_earlyStopping

import torch.nn.functional as F
from model import  ConvNet_baseNonlinearHead_SEED_saveFea
import random

parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--gpu-index', default=7, type=int, help='Gpu index.')
parser.add_argument('--timeLen', default=30, type=int,
                    help='time length in seconds')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)
torch.manual_seed(args.randSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(8)
args.device = torch.device('cuda')
torch.cuda.set_device(args.gpu_index)


timeLen = 1
timeStep = 1
fs = 200
channel_norm = False
time_norm = False
data_len = fs * timeLen

n_spatialFilters = 16
n_timeFilters = 16
timeFilterLen = 48
n_channs = 62
multiFact = 2

randomInit = False
stratified = []
data_dir = '/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref'
save_dir = './runs_seed/sepVideo_raw_batch9_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel'

data_train_all, data_val_all, label_train_repeat, label_val_repeat, n_samples, n_samples_remain = load_seed_raw_trainVal(
    data_dir, timeLen, timeStep, channel_norm)

print('data shape', data_train_all.shape, data_val_all.shape)

    
n_total_train = int(np.sum(n_samples[:9]))
n_total_val = int(np.sum(n_samples[9:]))
print('n_total', n_total_train, n_total_val)
print('n_samples', n_samples)
print('n_samples_remain', n_samples_remain)


bn = 1
print('batch size', bn)

n_folds = 5
n_per = 9
n_subs = 45
for fold in range(n_folds):
    print(fold)
    # Define model
    model = ConvNet_baseNonlinearHead_SEED_saveFea(n_spatialFilters, n_timeFilters, timeFilterLen, 
                                        n_channs, stratified, multiFact).to(args.device)
    print(model)
    para_num = sum([p.data.nelement() for p in model.parameters()])
    print('Total number of parameters:', para_num)

    # load pretrained parameters
    if not randomInit:
        with open(os.path.join(save_dir, 'results_pretrain.pkl'), 'rb') as f:
            results_pretrain = pickle.load(f)

        best_pretrain_epoch = int(results_pretrain['best_epoch'][fold])
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(best_pretrain_epoch)
        print('load:', checkpoint_name)
        # print(save_dir)
        checkpoint = torch.load(os.path.join(save_dir, str(fold), checkpoint_name), map_location=args.device)
        state_dict = checkpoint['state_dict']
        
        model.load_state_dict(state_dict, strict=False)

    
    n_samples_train = n_samples[:9]
    n_samples_val = n_samples[9:]
    val_sub = np.arange(n_per*fold, n_per*(fold+1))
    train_sub = list(set(np.arange(n_subs)) - set(val_sub))
    data_mean = np.mean(np.mean(data_train_all[train_sub, :, :], axis=1), axis=0)
    data_var = np.mean(np.var(data_train_all[train_sub, :, :], axis=1), axis=0)

    for i in range(n_subs):
        data_train_all[i,:,:] = (data_train_all[i,:,:] - data_mean) / np.sqrt(data_var + 1e-5)
        data_val_all[i,:,:] = (data_val_all[i,:,:] - data_mean) / np.sqrt(data_var + 1e-5)

    all_sub = np.arange(45)
    features1_de_train = np.zeros((len(all_sub), n_total_train, n_timeFilters, n_spatialFilters))
    features1_de_val = np.zeros((len(all_sub), n_total_val, n_timeFilters, n_spatialFilters))
    n = 0
    for sub in all_sub:
        data_train = data_train_all[sub, :, :]
        data_val = data_val_all[sub, :, :]
        label_train = np.array(label_train_repeat)
        label_val = np.array(label_val_repeat)
        print(sub)
        # print('data label', data_val.shape, label_val.shape)
        # Prepare data
        valset = SEEDDataset_raw(data_val, label_val, timeLen, timeStep, n_samples_val, n_samples_remain[9:], fs, transform=None) 
        val_loader = DataLoader(dataset=valset, batch_size=bn, pin_memory=True, num_workers=8, shuffle=False)

        isFirst = True
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            
            out = model(x_batch)
            isFirst = False
            out = out.detach().cpu().numpy()
            
            de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))

            if (counter + 1) * bn < n_total_val:
                features1_de_val[n, counter * bn: (counter + 1) * bn, :, :] = de
            else:
                features1_de_val[n, counter * bn:, :, :] = de

        trainset = SEEDDataset_raw(data_train, label_train, timeLen, timeStep, n_samples_train, n_samples_remain[:9], fs, transform=None) 
        train_loader = DataLoader(dataset=trainset, batch_size=bn, pin_memory=True, num_workers=8, shuffle=False)

        isFirst = True
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            
            out = model(x_batch)
            isFirst = False
            out = out.detach().cpu().numpy()
            
            de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))

            if (counter + 1) * bn < n_total_train:
                features1_de_train[n, counter * bn: (counter + 1) * bn, :, :] = de
            else:
                features1_de_train[n, counter * bn:, :, :] = de
        n = n + 1

    features1_de_train = features1_de_train.reshape(len(all_sub), n_total_train, 256)
    features1_de_val = features1_de_val.reshape(len(all_sub), n_total_val, 256)

    de = {'de_train': features1_de_train, 'de_val': features1_de_val}
    sio.savemat(os.path.join(save_dir, str(fold), 'features1_de_1s_all_normTrain.mat'), de)

