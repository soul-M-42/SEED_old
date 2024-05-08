import numpy as np
import os
import scipy.io as sio

def LDS(sequence):
        # print(sequence.shape) # (30, 256)

        # sequence_new = np.zeros_like(sequence) # (30, 256)
        ave = np.mean(sequence, axis=0)  # [256,]
        u0 = ave
        X = sequence.transpose((1, 0))  # [256, 30]

        V0 = 0.01
        A = 1
        T = 0.0001
        C = 1
        sigma = 1

        [m, n] = X.shape  # (1, 30)
        P = np.zeros((m, n))  # (1, 1, 30) dia
        u = np.zeros((m, n))  # (1, 30)
        V = np.zeros((m, n))  # (1, 1, 30) dia
        K = np.zeros((m, n))  # (1, 1, 30)

        K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m,))
        u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
        V[:, 0] = (np.ones((m,)) - K[:, 0] * C) * V0

        for i in range(1, n):
            P[:, i - 1] = A * V[:, i - 1] * A + T
            K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
            u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
            V[:, i] = (np.ones((m,)) - K[:, i] * C) * P[:, i - 1]

        X = u

        return X.transpose((1, 0))


randSeed = 7
tf = 16
sf = 16

datadir = './runs_seed/raw_batch15_timeLen30_tf16_sf16_tfLen48_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_accSel'
print(datadir)
# n_total = 3394
n_total = 3407
augTime = 1
n_subs = 45
decay_rate = 0.990

print('decay rate', decay_rate)
for fold in range(5):
    print('\nfold', fold)
    featuredir = os.path.join(datadir, str(fold), 'features1_de_1s_all_normTrain_rnPreWeighted{:.3f}.mat'.format(decay_rate))
    print(featuredir)
    de = sio.loadmat(featuredir)['de']
    n_samples = sio.loadmat(os.path.join('./data/SEED', 'n_samples.mat'))['n_samples'][0]
    n_samples_cum = np.concatenate([[0], np.cumsum(n_samples)])

    print(de.shape)
    de_lds = np.zeros((n_subs, sf, n_total * augTime, tf))
    for sub in range(n_subs):
        print('sub', sub)
        for aug in range(1, augTime + 1):
            de_one = np.reshape(de[sub, (n_total * (aug - 1)):(n_total * aug), :], (n_total, sf, tf))
            de_one = np.transpose(de_one, (1, 0, 2))
            for i in range(len(n_samples)):
                de_one_vid = de_one[:, int(n_samples_cum[i]):int(n_samples_cum[i + 1]), :]
                # print(de_one_vid.shape)
                de_one_vid = np.transpose(de_one_vid,(1,0,2))
                de_one_vid = de_one_vid.reshape((de_one_vid.shape[0], sf*tf))
                de_one_vid_lds = LDS(de_one_vid)  # 你需要自己实现 lds 函数
                # print(de_one_vid_lds.shape)
                de_one_vid_lds = de_one_vid_lds.reshape((de_one_vid_lds.shape[0], sf, tf))
                de_one_vid_lds = np.transpose(de_one_vid_lds, (1,0,2))
                de_lds[sub, :, (n_total * (aug - 1) + int(n_samples_cum[i])):(n_total * (aug - 1) + int(n_samples_cum[i + 1])), :] = de_one_vid_lds

    de_lds = np.reshape(np.transpose(de_lds, (0, 2, 1, 3)), (n_subs, n_total * augTime, sf * tf))
    np.save(os.path.join(datadir, str(fold), 'features1_de_1s_all_normTrain_rnPreWeighted{:.3f}_lds.npy'.format(decay_rate)), de_lds)
    # sio.savemat(os.path.join(datadir, str(fold), 'features1_de_1s_all_normTrain_rnPreWeighted{:.3f}_lds.npy'.format(decay_rate)), de_lds)

    import numpy as np

