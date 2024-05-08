import numpy as np
import scipy.io as sio
from io_utils import smooth_moving_average
import os

def load_seed_de(data_dir, channel_norm, prepSxk, isLds, isFilt, filtLen):
    n_subs = 45
    n_samples = sio.loadmat('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples.mat')['n_samples'][0]
    if prepSxk:
        if isLds:
            print('use lds smoothed de features prepared by sxk')
            data = sio.loadmat(data_dir)['de_lds']
        else:
            data = np.load('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/de.npy')
        data = data.transpose(0,2,3,1) # change to (n_subs, n_points, n_bands, n_channs)
        data = data.reshape(data.shape[0], data.shape[1], -1)
    else:
        data = np.load('/mnt/shenxinke/SEED/data_deLDS.npy')

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
    if isFilt:
        data = data.transpose(0,2,1)
        # print(data.shape)
        # Smoothing the data
        for i in range(n_subs):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])] = smooth_moving_average(data[
                                            i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])], filtLen)
        data = data.transpose(0,2,1)
    # print(data.shape)

    # Normalization for each sub
    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / np.std(data[i,:,:], axis=0)

    label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]] * n_samples[i]
    return data, label_repeat, n_samples


def load_seed_de_trainVal(data_dir, channel_norm, prepSxk, isLds, isFilt, filtLen):
    n_subs = 45
    n_samples = sio.loadmat('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples.mat')['n_samples'][0]
    if prepSxk:
        if isLds:
            print('use lds smoothed de features prepared by sxk')
            data = sio.loadmat(data_dir)['de_lds']
        else:
            data = np.load('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/de.npy')
        data = data.transpose(0,2,3,1) # change to (n_subs, n_points, n_bands, n_channs)
        data = data.reshape(data.shape[0], data.shape[1], -1) # (n_subs, n_points, n_feas)
    else:
        data = np.load('/mnt/shenxinke/SEED/data_deLDS.npy')

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
    if isFilt:
        data = data.transpose(0,2,1)
        # print(data.shape)
        # Smoothing the data
        for i in range(n_subs):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])] = smooth_moving_average(data[
                                            i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])], filtLen)
        data = data.transpose(0,2,1)
    print(data.shape)
    data_train = data[:, :int(np.sum(n_samples[:9])), :]
    data_val = data[:, int(np.sum(n_samples[:9])):, :]

    # Normalization for each sub
    if channel_norm:
        for i in range(data_train.shape[0]):
            data_train[i,:,:] = (data_train[i,:,:] - np.mean(data_train[i,:,:], axis=0)) / np.std(data_train[i,:,:], axis=0)
        for i in range(data_val.shape[0]):
            data_val[i,:,:] = (data_val[i,:,:] - np.mean(data_val[i,:,:], axis=0)) / np.std(data_val[i,:,:], axis=0)

    label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)
    label_train = label[:9]
    label_val = label[9:]

    label_repeat_train = []
    for i in range(len(label_train)):
        label_repeat_train = label_repeat_train + [label_train[i]]*n_samples[i]

    label_repeat_val = []
    for i in range(len(label_val)):
        label_repeat_val = label_repeat_val + [label_val[i]]*n_samples[i+9]
    return data_train, data_val, label_repeat_train, label_repeat_val, n_samples
    

def load_seed_pretrainFeat(datadir, channel_norm, timeLen, timeStep, isFilt, filtLen):
    # n_samples = sio.loadmat('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples.mat')['n_samples'][0]
    n_samples = sio.loadmat(os.path.join('./data/SEED', 'n_samples.mat'))['n_samples'][0]
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    if datadir[-4:] == '.npy':
        data = np.load(datadir)
        data[data < -10] = -5
    elif datadir[-4:] == '.mat':
        data = sio.loadmat(datadir)['de_lds']
        data[np.isnan(data)] = -8
        # data[data < -8] = -8
    if(np.isnan(data).any()):
        print('nan in data')
        data = np.nan_to_num(data, nan=0)
    print(data.shape)
    print(np.min(data), np.median(data))

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
    if isFilt:
        print('filtLen', filtLen)
        data = data.transpose(0,2,1)
        for i in range(data.shape[0]):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])] = smooth_moving_average(data[
                                            i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])], filtLen)
        data = data.transpose(0,2,1)

    # Normalization for each sub
    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / (np.std(data[i,:,:], axis=0) + 1e-3)

    label = [1,	0,	-1,	-1,	0,	1,	-1,	0,	1,	1,	0,	-1,	0,	1,	-1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*int(n_samples[i])
    return data, label_repeat, n_samples


def load_seed_pretrainFeat_trainVal(data_dir_train, data_dir_val, channel_norm, timeLen, timeStep):
    n_samples = sio.loadmat('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples.mat')['n_samples'][0]
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    data_train = sio.loadmat(data_dir_train)['de_lds']
    data_train[np.isnan(data_train)] = -8
    # data[data < -8] = -8

    data_val= sio.loadmat(data_dir_val)['de_lds']
    data_val[np.isnan(data_val)] = -8
    
    # data_use = data[:, np.max(data, axis=0)>1e-6]
    # data = data.reshape(45, int(np.sum(n_samples)), 256)
    print(data_train.shape, data_val.shape)
    print(np.min(data_train), np.median(data_train), np.max(data_train))
    print(np.min(data_val), np.median(data_val), np.max(data_val))

    # Normalization for each sub
    if channel_norm:
        for i in range(data_train.shape[0]):
            data_train[i,:,:] = (data_train[i,:,:] - np.mean(data_train[i,:,:], axis=0)) / (np.std(data_train[i,:,:], axis=0) + 1e-3)
        for i in range(data_val.shape[0]):
            data_val[i,:,:] = (data_val[i,:,:] - np.mean(data_val[i,:,:], axis=0)) / (np.std(data_val[i,:,:], axis=0) + 1e-3)

    label = [1,	0,	-1,	-1,	0,	1,	-1,	0,	1,	1,	0,	-1,	0,	1,	-1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return data_train, data_val, label_repeat, n_samples


def load_seed_pretrainFeat_trainVal_sepVideo(data_dir_train, data_dir_val, channel_norm, timeLen, timeStep):
    n_samples = sio.loadmat('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples.mat')['n_samples'][0]
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    data_train = sio.loadmat(data_dir_train)['de_lds']
    data_train[np.isnan(data_train)] = -8
    # data[data < -8] = -8

    data_val= sio.loadmat(data_dir_val)['de_lds']
    data_val[np.isnan(data_val)] = -8
    
    # data_use = data[:, np.max(data, axis=0)>1e-6]
    # data = data.reshape(45, int(np.sum(n_samples)), 256)
    print(data_train.shape, data_val.shape)
    print(np.min(data_train), np.median(data_train), np.max(data_train))
    print(np.min(data_val), np.median(data_val), np.max(data_val))

    # Normalization for each sub
    if channel_norm:
        for i in range(data_train.shape[0]):
            data_train[i,:,:] = (data_train[i,:,:] - np.mean(data_train[i,:,:], axis=0)) / (np.std(data_train[i,:,:], axis=0) + 1e-3)
        for i in range(data_val.shape[0]):
            data_val[i,:,:] = (data_val[i,:,:] - np.mean(data_val[i,:,:], axis=0)) / (np.std(data_val[i,:,:], axis=0) + 1e-3)

    label = [1,	0,	-1,	-1,	0,	1,	-1,	0,	1,	1,	0,	-1,	0,	1,	-1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)
    label_train = label[:9]
    label_val = label[9:]

    label_repeat_train = []
    for i in range(len(label_train)):
        label_repeat_train = label_repeat_train + [label_train[i]]*n_samples[i]

    label_repeat_val = []
    for i in range(len(label_val)):
        label_repeat_val = label_repeat_val + [label_val[i]]*n_samples[i+9]
    return data_train, data_val, label_repeat_train, label_repeat_val, n_samples


def load_seed_raw(data_dir, timeLen, timeStep, channel_norm, time_norm):
    n_samples_old = sio.loadmat(os.path.join(data_dir, 'n_samples.mat'))['n_samples'][0]
    data = np.load(os.path.join(data_dir, 'data_all.npy'))
    print('data loaded:', data.shape)
    
    # (n_subs, n_points, n_channs)
    data = data.transpose(0,2,1)
    n_samples_remain = np.zeros(len(n_samples_old), dtype=int)
    n_samples = np.zeros(len(n_samples_old), dtype=int)
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples_old[i] - timeLen) / timeStep + 1)
        n_samples_remain[i] = n_samples_old[i] - n_samples[i] * timeStep
        
    print('n_samples sum', np.sum(n_samples))
    print('n_samples_old', n_samples_old)

    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / np.std(data[i,:,:], axis=0)

    if time_norm:
        data = (data - np.tile(np.expand_dims(np.mean(data, axis=2), 2), (1, 1, data.shape[2]))) / np.tile(
            np.expand_dims(np.std(data, axis=2), 2), (1, 1, data.shape[2])
        )

    label = [1,	0,	-1,	-1,	0,	1,	-1,	0,	1,	1,	0,	-1,	0,	1,	-1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return data, label_repeat, n_samples, n_samples_remain

def load_Faced_raw(data_dir, timeLen, timeStep, channel_norm, time_norm, n_class='3'):
    n_samples_old = sio.loadmat(os.path.join(data_dir, 'n_samples.mat'))['n_samples'][0]
    data = np.load(os.path.join(data_dir, 'data_all.npy'))
    print('data loaded:', data.shape)
    
    # (n_subs, n_points, n_channs)
    data = data.transpose(0,2,1)
    n_samples_remain = np.zeros(len(n_samples_old), dtype=int)
    n_samples = np.zeros(len(n_samples_old), dtype=int)
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples_old[i] - timeLen) / timeStep + 1)
        n_samples_remain[i] = n_samples_old[i] - n_samples[i] * timeStep
        
    print('n_samples sum', np.sum(n_samples))
    print('n_samples_old', n_samples_old)

    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / np.std(data[i,:,:], axis=0)

    if time_norm:
        data = (data - np.tile(np.expand_dims(np.mean(data, axis=2), 2), (1, 1, data.shape[2]))) / np.tile(
            np.expand_dims(np.std(data, axis=2), 2), (1, 1, data.shape[2])
        )

    Faced_Labels_3 = np.array([0] * 12 + [1] * 4 + [2] * 12)
    Faced_Labels_9 = np.array([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 4 + [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3)
    if(n_class == '3'):
        label = Faced_Labels_3
    elif(n_class == '9'):
        label = Faced_Labels_9
    print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return data, label_repeat, n_samples, n_samples_remain


def load_seed_raw_trainVal(data_dir, timeLen, timeStep, channel_norm):
    n_samples_old = sio.loadmat(os.path.join(data_dir, 'n_samples.mat'))['n_samples'][0]
    data = np.load(os.path.join(data_dir, 'data_all.npy'))
    print('data loaded:', data.shape) 
    
    data = data.transpose(0,2,1)    # change to (n_subs, n_points, n_channs)
    n_samples_remain = np.zeros(len(n_samples_old), dtype=int)
    n_samples = np.zeros(len(n_samples_old), dtype=int)
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples_old[i] - timeLen) / timeStep + 1)
        n_samples_remain[i] = n_samples_old[i] - n_samples[i] * timeStep
        
    print('n_samples sum', np.sum(n_samples))
    print('n_samples_old', n_samples_old)

    # print(n_samples_old[:9].sum()*200)
    data_train = data[:, :int(n_samples_old[:9].sum()*200), :]
    data_val = data[:, int(n_samples_old[:9].sum()*200):, :]

    if channel_norm:
        for i in range(data_train.shape[0]):
            data_train[i,:,:] = (data_train[i,:,:] - np.mean(data_train[i,:,:], axis=0)) / np.std(data_train[i,:,:], axis=0)
        for i in range(data_val.shape[0]):
            data_val[i,:,:] = (data_val[i,:,:] - np.mean(data_val[i,:,:], axis=0)) / np.std(data_val[i,:,:], axis=0)

    label = [1,	0,	-1,	-1,	0,	1,	-1,	0,	1,	1,	0,	-1,	0,	1,	-1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)
    label_train = label[:9]
    label_val = label[9:]

    label_repeat_train = []
    for i in range(len(label_train)):
        label_repeat_train = label_repeat_train + [label_train[i]]*n_samples[i]
    label_repeat_val = []
    for i in range(len(label_val)):
        label_repeat_val = label_repeat_val + [label_val[i]]*n_samples[i+9]
    return data_train, data_val, label_repeat_train, label_repeat_val, n_samples, n_samples_remain


def load_seed_de_trainVal_intraSub(data_sub, channel_norm):
    n_samples = sio.loadmat('/mnt/shenxinke/SEED/interp_removeAber_filt4_47_reref/n_samples.mat')['n_samples'][0]

    print(data_sub.shape)
    # if channel_norm:
    #     data_sub = (data_sub - np.mean(data_sub, axis=0)) / np.std(data_sub, axis=0)
    data_train = data_sub[:int(np.sum(n_samples[:9])), :]
    data_val = data_sub[int(np.sum(n_samples[:9])):, :]

    # Normalization for each sub
    if channel_norm:
        data_train = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
        data_val = (data_val - np.mean(data_val, axis=0)) / np.std(data_val, axis=0)

    label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    for i in range(len(label)):
        label[i] = label[i] + 1
    print(label)
    label_train = label[:9]
    label_val = label[9:]

    label_repeat_train = []
    for i in range(len(label_train)):
        label_repeat_train = label_repeat_train + [label_train[i]]*n_samples[i]

    label_repeat_val = []
    for i in range(len(label_val)):
        label_repeat_val = label_repeat_val + [label_val[i]]*n_samples[i+9]
    return data_train, data_val, label_repeat_train, label_repeat_val, n_samples