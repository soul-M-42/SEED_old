import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader
from io_utils import smooth_moving_average, SEEDDataset_raw
from load_data import load_seed_raw
from simCLR import SimCLR
from train_utils import train_earlyStopping

import torch.nn as nn
import torch.nn.functional as F
from model import stratified_layerNorm
import random

parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--gpu-index', default=6, type=int, help='Gpu index.')
parser.add_argument('--timeLen', default=30, type=int,
                    help='time length in seconds')
parser.add_argument('--learning-rate', default=0.0007, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', default=0.015, type=float,
                    metavar='W', help='weight decay (default: 0.05)',
                    dest='weight_decay')
parser.add_argument('--randSeed', default=0, type=int,
                    help='random seed')
parser.add_argument('--timeFilterLen', default=48, type=int,
                    help='time filter length')
parser.add_argument('--n_spatialFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--n_timeFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--multiFact', default=2, type=int,
                    help='time filter length')
# parser.add_argument('--batch-size-pretrain', default=2959, type=int, metavar='N',
#                     help='mini-batch size')
args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)
torch.manual_seed(args.randSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(8)
args.device = torch.device('cuda')

timeLen = 1
timeStep = 1
fs = 200
channel_norm = False
time_norm = False
data_len = fs * timeLen

n_spatialFilters = args.n_spatialFilters
n_timeFilters = args.n_timeFilters
timeFilterLen = args.timeFilterLen
n_channs = 62
multiFact = 2

randomInit = False
stratified = ['initial', 'middle1']
data_dir = '/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref'
# save_dir = '/mnt/shenxinke/SEED/branch5_setSeed/runs_seed/raw_batch15_timeLen30_tf%d_sf%d_tfLen%d_multiFact2_lr%f_wd%f_epochs100_randSeed%d_accSel' % (
#     n_timeFilters, n_spatialFilters, timeFilterLen, args.learning_rate, args.weight_decay, args.randSeed)
save_dir = '/mnt/shenxinke/SEED/branch5_setSeed/runs_seed/baseline_simseqclr_timeLen30_bs_mask_shift_gaus_scale_pre100_noStrat'
# save_dir = '/mnt/shenxinke/SEED/branch5/runs_seed/raw_newPre_timeLen%d_multiFact%d_lr0.000700_epochs100' % (args.timeLen, multiFact)
print(save_dir)

class TrainSampler_sub_noShuffle():
    def __init__(self, n_subs_all, n_samples, batch_size, n_subs=1):
        self.n_subs = n_subs
        self.n_subs_all = n_subs_all
        # Number of data points per session
        self.batch_size = batch_size
        self.n_samples_sum = int(np.sum(n_samples))
        self.n_samples_per_sub = int(batch_size / n_subs)

        ind_all = np.zeros((n_subs_all, self.n_samples_sum))
        for i in range(n_subs_all):
            tmp = np.arange(self.n_samples_sum) + i * self.n_samples_sum
#             np.random.shuffle(tmp)
            ind_all[i,:] = tmp
#         np.random.shuffle(ind_all)
        self.ind_all = ind_all

        # self.n_times = int(n_subs_all * self.n_samples_sum // batch_size)
        self.n_times_sub = int(n_subs_all / n_subs)
        self.n_times_vid = int(self.n_samples_sum / self.n_samples_per_sub)


    def __len__(self):
        return self.n_times_sub * self.n_times_vid

    def __iter__(self):
        for i in range(self.n_times_vid):
            for j in range(self.n_times_sub):
                ind_sel = self.ind_all[j*self.n_subs: (j+1)*self.n_subs, self.n_samples_per_sub*i: self.n_samples_per_sub*(i+1)]
                ind_sel = ind_sel.reshape(-1)
                batch = torch.LongTensor(ind_sel)
                # print(batch)
                yield batch

class ConvNet_baseNonlinearHead_SEED(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact):
        super(ConvNet_baseNonlinearHead_SEED, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.avgpool = nn.AvgPool2d((1, 24))
        self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*multiFact, (n_spatialFilters, 1), groups=n_timeFilters)
        self.timeConv2 = nn.Conv2d(n_timeFilters*multiFact, n_timeFilters*multiFact*multiFact, (1, 4), groups=n_timeFilters*multiFact)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified
    def forward(self, input):
        # print(input.shape)
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, input.shape[0])

        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        out1 = out.clone()
        out = F.elu(out)
        out = self.avgpool(out)

        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, out.shape[0])

        out = F.elu(self.spatialConv2(out))
        out = F.elu(self.timeConv2(out))
        # print(out.shape)

        if 'middle2' in self.stratified:
            out = stratified_layerNorm(out, out.shape[0])
        return out, out1



data_train, label_repeat, n_samples, n_samples_remain = load_seed_raw(data_dir, timeLen, timeStep, channel_norm, time_norm)
torch.cuda.set_device(args.gpu_index)

n_total = int(np.sum(n_samples))
print('n_total', n_total)
print('n_samples', n_samples)
print('n_samples_remain', n_samples_remain)

n_folds = 5
for fold in range(n_folds):
    features1_de = np.zeros((45, n_total, n_timeFilters, n_spatialFilters))
    # features1 = np.zeros((45, n_total, 16, 16))
    # features = np.zeros((45, n_total, n_timeFilters*multiFact*multiFact))
    print('fold', fold)

    model = ConvNet_baseNonlinearHead_SEED(n_spatialFilters, n_timeFilters, timeFilterLen, 
                                      n_channs, stratified, multiFact).to(args.device)
    print(model)
    para_num = sum([p.data.nelement() for p in model.parameters()])
    print('Total number of parameters:', para_num)

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

    train_sub = np.arange(45)
    data_train = data_train.reshape(-1, data_train.shape[-1])
    label_train = np.tile(label_repeat, len(train_sub))
    # print('label_train length', len(label_train))
    
    trainset = SEEDDataset_raw(data_train, label_train, timeLen, timeStep, n_samples, n_samples_remain, fs)  
    train_sampler = TrainSampler_sub_noShuffle(len(train_sub), n_samples, n_total)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
    
    for counter, (x_batch, y_batch) in enumerate(train_loader):
        print('counter:', counter)
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)
        
        out, out1 = model(x_batch)
        out1 = out1.detach().cpu().numpy()
        out = out.detach().cpu().numpy()
        
        de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out1, 3)))
        features1_de[counter,:,:,:] = de
        # features1[counter, :, :, :] = np.var(out1, 3)
        # out = np.squeeze(out, 2)
        # features[counter,:,:] = np.mean(out, 2)
        # features[counter*n_total: (counter+1)*n_total,:] = np.squeeze(np.mean(out, 3), 2)
    features1_de = features1_de.reshape(45, n_total, n_timeFilters * n_spatialFilters)
    # features1 = features1.reshape(45, n_total, 256)

    de = {'de': features1_de}
    sio.savemat(os.path.join(save_dir, str(fold), 'features1_de_1s.mat'), de)

    # np.save(os.path.join(save_dir, str(fold), 'features1_de_1s.npy'), features1_de)
    # np.save(os.path.join(save_dir, str(fold), 'features1_1s.npy'), features1)
    # np.save(os.path.join(save_dir, str(fold), 'features_1s.npy'), features)

