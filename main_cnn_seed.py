import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader
from io_utils import SEEDDataset_raw, TrainSampler_SEED, TrainSampler_video
from load_data import load_seed_raw
from model import ConvNet_baseNonlinearHead_SEED, ConvNet_nonlinearHead_norm_SEED
from simCLR import SimCLR
from train_utils import train_earlyStopping
from torchvision import transforms
import random

parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')

parser.add_argument('--gpu-index', default=5, type=int, help='Gpu index.')

parser.add_argument('--epochs-pretrain', default=100, type=int, metavar='N',
                    help='number of total epochs to run in pretraining')
parser.add_argument('--restart_times', default=3, type=int, metavar='N',
                    help='number of total epochs to run in pretraining')
parser.add_argument('--max-tol-pretrain', default=30, type=int, metavar='N',
                    help='number of max tolerence for epochs with no val loss decrease in pretraining')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help=' n views in contrastive learning')
parser.add_argument('--batch-size-pretrain', default=15, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.0007, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', default=0.015, type=float,
                    metavar='W', help='weight decay (default: 0.05)',
                    dest='weight_decay')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-times', default=1, type=int,
                    help='number of sampling times for one sub pair (in one session)')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--timeLen', default=30, type=int,
                    help='time length in seconds')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')

parser.add_argument('--timeFilterLen', default=48, type=int,
                    help='time filter length')
parser.add_argument('--n_spatialFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--n_timeFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--multiFact', default=2, type=int,
                    help='time filter length')

args = parser.parse_args()


random.seed(args.randSeed)
np.random.seed(args.randSeed)
torch.manual_seed(args.randSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(8)
stratified = ['initial', 'middle1', 'middle2']
channel_norm = False
time_norm = False

n_spatialFilters = args.n_spatialFilters
n_timeFilters = args.n_timeFilters
timeFilterLen = args.timeFilterLen
multiFact = 2
n_channs = 62
fs = 200

# The current method only supports timeLen and timeStep that can 整除. So timeStep would be better 1.
timeLen = args.timeLen
timeStep = 15
data_len = fs * timeLen

for pos in stratified:
    assert pos in ['initial', 'middle1', 'middle2', 'final', 'final_batch', 'middle1_batch', 'middle2_batch', 'no']

label_init = [1,	0,	-1,	-1,	0,	1,	-1,	0,	1,	1,	0,	-1,	0,	1,	-1]
for i in range(len(label_init)):
    label_init[i] = label_init[i] + 1
print(label_init)

data_dir = '/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/'
data, label_repeat, n_samples, n_samples_remain = load_seed_raw(data_dir, timeLen, timeStep, channel_norm, time_norm)

args.device = torch.device('cuda')
torch.cuda.set_device(args.gpu_index)

# We use five-fold cross-validation in contrastive learning
n_folds = 5

args.log_dir = 'raw_batch%d_timeLen%d_tf%d_sf%d_tfLen%d_multiFact%d_lr%f_wd%f_epochs%d_randSeed%d_accSel' % (
    args.batch_size_pretrain, timeLen, n_timeFilters, n_spatialFilters, timeFilterLen,
    multiFact, args.learning_rate, args.weight_decay, args.epochs_pretrain, args.randSeed)
root_dir = './'
print(args)

save_dir = os.path.join(root_dir, 'runs_seed', args.log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

results_pretrain = {}
results_pretrain['train_top1_history'], results_pretrain['val_top1_history'] = np.zeros((n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
results_pretrain['train_top5_history'], results_pretrain['val_top5_history'] =np.zeros((n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
results_pretrain['train_loss_history'], results_pretrain['val_loss_history'] = np.zeros((n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
results_pretrain['best_val_top1'], results_pretrain['best_val_top5'] = np.zeros(n_folds), np.zeros(n_folds)
results_pretrain['best_val_loss'], results_pretrain['best_epoch'] = np.zeros(n_folds), np.zeros(n_folds)

for fold in range(n_folds):
    print('fold', fold)

    model_pre = ConvNet_baseNonlinearHead_SEED(n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified=stratified, multiFact=multiFact,
                                                args=args).to(args.device)
    print(model_pre)
    para_num = sum([p.data.nelement() for p in model_pre.parameters()])
    print('Total number of parameters:', para_num)

    val_sub = np.arange(9*fold, 9*(fold+1))
    print('val', val_sub)
    train_sub = np.array(list(set(np.arange(45)) - set(val_sub)))
    print('train', train_sub)

    data_train = data[list(train_sub), :, :].reshape(-1, data.shape[-1])
    label_train = np.tile(label_repeat, len(train_sub))
    print('label_train length', len(label_train))

    data_val = data[list(val_sub), :, :].reshape(-1, data.shape[-1])
    label_val = np.tile(label_repeat, len(val_sub))
    print('label_val length', len(label_val))

    trainset = SEEDDataset_raw(data_train, label_train, timeLen, timeStep, n_samples, n_samples_remain, fs, transform=None)
    valset = SEEDDataset_raw(data_val, label_val, timeLen, timeStep, n_samples, n_samples_remain, fs)

    print('n_samples', n_samples)
    print('sample across subjects')
    train_sampler = TrainSampler_SEED(len(train_sub), n_times=args.n_times, batch_size=args.batch_size_pretrain,
                                            n_samples=n_samples)
    val_sampler = TrainSampler_SEED(len(val_sub), n_times=args.n_times, batch_size=args.batch_size_pretrain,
                                        n_samples=n_samples)

    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)

    optimizer = torch.optim.Adam(model_pre.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs_pretrain // args.restart_times,
                                                                        eta_min=0,
                                                                        last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(args=args, model=model_pre, optimizer=optimizer, scheduler=scheduler,
                        log_dir=os.path.join(save_dir, str(fold)), stratified='no')
        model_pre, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history = simclr.train(
            train_loader, val_loader)

    results_pretrain['train_top1_history'][fold,:], results_pretrain['val_top1_history'][fold,:] = train_top1_history, val_top1_history
    results_pretrain['train_top5_history'][fold,:], results_pretrain['val_top5_history'][fold,:] = train_top5_history, val_top5_history
    results_pretrain['train_loss_history'][fold,:], results_pretrain['val_loss_history'][fold,:] = train_loss_history, val_loss_history
    results_pretrain['best_val_top1'][fold] = results_pretrain['val_top1_history'][fold, best_epoch]
    results_pretrain['best_val_top5'][fold] = results_pretrain['val_top5_history'][fold, best_epoch]
    results_pretrain['best_val_loss'][fold] = results_pretrain['val_loss_history'][fold, best_epoch]
    results_pretrain['best_epoch'][fold] = best_epoch

    np.save(os.path.join(save_dir, str(fold), 'train_top1_history.npy'), train_top1_history)
    np.save(os.path.join(save_dir, str(fold), 'val_top1_history.npy'), val_top1_history)
    np.save(os.path.join(save_dir, str(fold), 'train_top5_history.npy'), train_top5_history)
    np.save(os.path.join(save_dir, str(fold), 'val_top5_history.npy'), val_top5_history)
    np.save(os.path.join(save_dir, str(fold), 'train_loss_history.npy'), train_loss_history)
    np.save(os.path.join(save_dir, str(fold), 'val_loss_history.npy'), val_loss_history)

with open(os.path.join(save_dir, 'results_pretrain.pkl'), 'wb') as f:
    pickle.dump(results_pretrain, f)
print(save_dir)
print('val loss mean: %.3f, std: %.3f; val acc top1 mean: %.3f, std: %.3f; val acc top5 mean: %.3f, std: %.3f' % (
    np.mean(results_pretrain['best_val_loss']), np.std(results_pretrain['best_val_loss']),
    np.mean(results_pretrain['best_val_top1']), np.std(results_pretrain['best_val_top1']),
    np.mean(results_pretrain['best_val_top5']), np.std(results_pretrain['best_val_top5'])))