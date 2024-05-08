import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader
from io_utils import DEDataset, TrainSampler_sub
from load_data import load_seed_pretrainFeat, load_seed_de
from model import simpleNN3
from train_utils import train_earlyStopping
import random

parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--epochs-finetune', default=100, type=int, metavar='N',
                    help='number of total epochs to run in finetuning')
parser.add_argument('--max-tol', default=30, type=int, metavar='N',
                    help='number of max tolerence for epochs with no val loss decrease in finetuning')
parser.add_argument('--batch-size-finetune', default=270, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-finetune', default=0.0005, type=float, metavar='LR',
                    help='learning rate in finetuning')
parser.add_argument('--weight-decay', default=0.025, type=float,
                    metavar='W', help='weight decay (default: 0.025)',
                    dest='weight_decay')
parser.add_argument('--hidden-dim', default=30, type=int, metavar='N',
                    help='hidden dim of the NN')

parser.add_argument('--gpu-index', default=6, type=int, help='Gpu index.')

parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)
torch.manual_seed(args.randSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(8)
use_features = 'features1_de_1s_all_normTrain_rnPreWeighted0.990_lds.npy'
isLds = True
channel_norm = False
time_norm = False
stratified = False
isFilt = False

hidden_dim = args.hidden_dim
# fs = 200
fs = 125

timeLen = 1
timeStep = 1
filtLen = 1

args.device = torch.device('cuda')
torch.cuda.set_device(args.gpu_index)

save_dir = './runs_seed/raw_batch15_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel'
print(args)
print(save_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n_folds = 15
n_per = int(45 / n_folds)


print('start finetuning')

results_finetune = {}
results_finetune['train_loss_history'], results_finetune['val_loss_history'] = np.zeros((n_folds, args.epochs_finetune)), np.zeros((n_folds, args.epochs_finetune))
results_finetune['train_acc_history'], results_finetune['val_acc_history'] = np.zeros((n_folds, args.epochs_finetune)), np.zeros((n_folds, args.epochs_finetune))
results_finetune['best_val_acc'], results_finetune['best_val_loss'] = np.zeros(n_folds), np.zeros(n_folds)
results_finetune['best_epoch'] = np.zeros(n_folds)
results_finetune['best_confusion'] = np.zeros((n_folds, 3, 3))


save_dir_ft = os.path.join(save_dir, 'wd%.3f' % args.weight_decay)
if not os.path.exists(save_dir_ft):
    os.makedirs(save_dir_ft)

for fold in range(n_folds):
    print('fold', fold)
    args.save_dir_ft = os.path.join(save_dir_ft, str(fold))
    if not os.path.exists(args.save_dir_ft):
        os.makedirs(args.save_dir_ft)

    pre_fold = int(fold // 3)
    print('pretrain fold', pre_fold)
    data_dir = os.path.join(save_dir, str(pre_fold), use_features)
    print(data_dir)
    data, label_repeat, n_samples = load_seed_pretrainFeat(data_dir, channel_norm, timeLen, timeStep, isFilt, filtLen)
    print('data loaded:', data.shape)

    val_sub = np.arange(n_per*fold, n_per*(fold+1))
    print('val', val_sub)
    train_sub = np.array(list(set(np.arange(45)) - set(val_sub)))
    print('train', train_sub)

    data_train = data[list(train_sub), :, :].reshape(-1, data.shape[-1])
    label_train = np.tile(label_repeat, len(train_sub))
    print(data_train.shape, label_train.shape)
    print('label_train length', len(label_train))

    data_val = data[list(val_sub), :, :].reshape(-1, data.shape[-1])
    label_val = np.tile(label_repeat, len(val_sub))
    print('label_val length', len(label_val))

    trainset = DEDataset(data_train, label_train)
    valset = DEDataset(data_val, label_val)

    print(n_samples)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size_finetune, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size_finetune, shuffle=False, num_workers=8) 


    inp_dim = data_train.shape[-1]
    model = simpleNN3(inp_dim, hidden_dim, 3).to(args.device)
    print(model)
    para_num = sum([p.data.nelement() for p in model.parameters()])
    print('Total number of parameters:', para_num)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_finetune, weight_decay=args.weight_decay)

    print(save_dir_ft)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs_finetune, gamma=0.8, last_epoch=-1, verbose=False)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_confusion = train_earlyStopping(
        args, train_loader, val_loader, model, criterion, optimizer, scheduler)

    results_finetune['train_loss_history'][fold,:], results_finetune['val_loss_history'][fold,:] = train_loss_history, val_loss_history
    results_finetune['train_acc_history'][fold,:], results_finetune['val_acc_history'][fold,:] = train_acc_history, val_acc_history
    results_finetune['best_val_acc'][fold] = results_finetune['val_acc_history'][fold, best_epoch]
    results_finetune['best_val_loss'][fold] = results_finetune['val_loss_history'][fold, best_epoch]
    results_finetune['best_epoch'][fold] = best_epoch
    results_finetune['best_confusion'][fold, :, :] = best_confusion


    with open(os.path.join(save_dir_ft, 'results_finetune.pkl'), 'wb') as f:
        pickle.dump(results_finetune, f)
print(save_dir_ft)
print(args)

print('val loss mean: %.3f, std: %.3f; val acc mean: %.3f, std: %.3f' % (
    np.mean(results_finetune['best_val_loss']), np.std(results_finetune['best_val_loss']),
    np.mean(results_finetune['best_val_acc']), np.std(results_finetune['best_val_acc']))
)