import os
import scipy.io as sio
import numpy as np

save_dir = './runs_seed/sepVideo_raw_batch9_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel'
print(save_dir)
bn_val = 1
# momentum = 0.9
data_dir = '/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/'
n_samples = sio.loadmat(os.path.join(data_dir, 'n_samples.mat'))['n_samples'][0]
n_total_train = np.sum(n_samples[:9])
n_total_val = np.sum(n_samples[9:])

n_counters_train = int(np.ceil(n_total_train / bn_val))
n_counters_val = int(np.ceil(n_total_val / bn_val))

use_features = 'features1_de_1s_all_normTrain'
print(use_features)

n_subs = 45
n_folds = 5
n_per = 9

decay_rate = 0.99

for fold in range(n_folds):
    print(fold)
    data = sio.loadmat(os.path.join(save_dir, str(fold), use_features+'.mat'))
    data_train = data['de_train']
    data_train[np.isnan(data_train)] = -30
    data_val = data['de_val']
    data_val[np.isnan(data_val)] = -30
    print(data_train.shape, data_val.shape)

    val_sub = np.arange(n_per*fold, n_per*(fold+1))
    train_sub = list(set(np.arange(n_subs)) - set(val_sub))
    data_mean = np.mean(np.mean(data_train[train_sub, :, :], axis=1), axis=0)
    data_var = np.mean(np.var(data_train[train_sub, :, :], axis=1), axis=0)
    
    data_norm_train = np.zeros_like(data_train)
    for sub in range(data_train.shape[0]):
        running_sum = np.zeros(data_train.shape[-1])
        running_square = np.zeros(data_train.shape[-1])
        decay_factor = 1
        for counter in range(n_counters_train):
            data_one = data_train[sub, counter*bn_val: (counter+1)*bn_val, :]
            running_sum = running_sum + data_one
            running_mean = running_sum / (counter+1)
            running_square = running_square + data_one**2
            running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

            curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
            curr_var = decay_factor*data_var + (1-decay_factor)*running_var
            decay_factor = decay_factor*decay_rate

            data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
            data_norm_train[sub, counter*bn_val: (counter+1)*bn_val, :] = data_one

    data_norm_val = np.zeros_like(data_val)
    for sub in range(data_val.shape[0]):
        running_sum = np.zeros(data_val.shape[-1])
        running_square = np.zeros(data_val.shape[-1])
        decay_factor = 1
        for counter in range(n_counters_val):
            data_one = data_val[sub, counter*bn_val: (counter+1)*bn_val, :]
            running_sum = running_sum + data_one
            running_mean = running_sum / (counter+1)
            running_square = running_square + data_one**2
            running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

            curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
            curr_var = decay_factor*data_var + (1-decay_factor)*running_var
            decay_factor = decay_factor*decay_rate

            data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
            data_norm_val[sub, counter*bn_val: (counter+1)*bn_val, :] = data_one

    de = {'de_train': data_norm_train, 'de_val': data_norm_val}

    sio.savemat(os.path.join(save_dir, str(fold), use_features+'_rnPreWeighted%.3f.mat' % decay_rate), de)
            

