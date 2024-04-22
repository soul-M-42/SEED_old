import torch.nn as nn
import torch.nn.functional as F
import torch

def stratified_norm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_str[n_samples*i: n_samples*(i+1), :] = (out[n_samples*i: n_samples*(i+1), :] - out[n_samples*i: n_samples*(i+1), :].mean(
            dim=0)) / (out[n_samples*i: n_samples*(i+1), :].std(dim=0) + 1e-3)
    return out_str

def batch_norm(out):
    out_str = out.clone()
    out_str = (out - out.mean(dim=0)) / (out.std(dim=0) + 1e-3)
    return out_str

def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        out_oneSub_str = out_oneSub.clone()
        # We don't care about the channels with very small activations
        # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
        #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(
            0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
    return out_str

def batch_layerNorm(out):
    n_samples, chn1, chn2, n_points = out.shape
    out = out.reshape(n_samples, -1, n_points).permute(0,2,1)
    out = out.reshape(n_samples*n_points, -1)
    out_str = (out - out.mean(dim=0)) / (out.std(dim=0) + 1e-3)
    out_str = out_str.reshape(n_samples, n_points, chn1*chn2).permute(
        0,2,1).reshape(n_samples, chn1, chn2, n_points)
    return out_str

class ConvNet_baseNonlinearHead_SEED(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact, args):
        super(ConvNet_baseNonlinearHead_SEED, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.avgpool = nn.AvgPool2d((1, 24))
        self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*multiFact, (n_spatialFilters, 1), groups=n_timeFilters)
        self.timeConv2 = nn.Conv2d(n_timeFilters*multiFact, n_timeFilters*multiFact*multiFact, (1, 4), groups=n_timeFilters*multiFact)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified
        self.args = args
    def forward(self, input):
        # print(input.shape)
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))

        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        out = F.elu(out)
        out = self.avgpool(out)

        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))

        out = F.elu(self.spatialConv2(out))
        out = F.elu(self.timeConv2(out))

        if 'middle2' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))

        out = out.reshape(out.shape[0], -1)
        return out


class ConvNet_baseNonlinearHead_SEED_saveFea(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact):
        super(ConvNet_baseNonlinearHead_SEED_saveFea, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.stratified = stratified
    def forward(self, input):
        # print(input.shape)
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, input.shape[0])
        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        return out


class simpleNN3(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(simpleNN3, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
