"""
为了加速实验，搞了这么一出
Author: Wang Rui / rwang
Version: 1.0

备注：
    __main__ 给出了训练和评估的案例

模型设计：
    替换测试案例中的说话人嵌入提取模型、顶层分类器模型
    相关的模块：ResNetSE34L, AMSoftmax

数据增益：
    替换测试案例中的文件读取方法
    相关的模型：loadWAV

测试案例：
```
if __name__ == "__main__":
    # 定义训练集、测试集及其两者的根目录
    trainlst = "/workspace/rwang/voxceleb/train_list.txt"
    testlst = "/workspace/rwang/VoxSRC2020/data/verif/trials.txt"
    traindir = "/workspace/rwang/voxceleb/voxceleb2/"
    testdir = "/workspace/rwang/voxceleb/"
    maptrain5994 = "/workspace/rwang/competition/voxsrc2020/maptrain5994.txt"
    # 载入训练集
    train = load_train(trainlst=trainlst, traindir=traindir,
                       maptrain5994=maptrain5994)
    # 载入测试集
    trial = load_trial(testlst=testlst, testdir=testdir)
    # 定义说话人嵌入提取模型
    net = ResNetSE34L(nOut=512, num_filters=[16, 32, 64, 128])
    # 定义顶层分类器模型
    top = AMSoftmax(in_feats=512, n_classes=5994, m=0.2, s=30)
    # sklearn 模型生成
    snet = SpeakerNet(net=net, top=top)
    # 模型训练
    modelst, step_num, loss, prec1, prec5 = snet.train(train, num_epoch=1)
    # 模型评估
    eer, thresh, all_scores, all_labels, all_trials, trials_feat = snet.eval(
        trial, step_num=0, trials_feat=None)
```
"""

import os
import glob
import math
import random
import IPython
import pandas as pd
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
np.random.seed(22)
torch.manual_seed(22)
torch.cuda.manual_seed(22)
torch.cuda.manual_seed_all(22)

# 数据准备
trainlst = "/workspace/rwang/voxceleb/train_list.txt"
testlst = "/workspace/rwang/VoxSRC2020/data/verif/trials.txt"
traindir = "/workspace/rwang/voxceleb/voxceleb2/"
testdir = "/workspace/rwang/voxceleb/"
maptrain5994 = "/workspace/rwang/competition/voxsrc2020/maptrain5994.txt"
batch_size = 200
num_worker = 4
max_utt_per_spk = 100
L = 32240
eval_L = 48240
# 模型准备
n_mel = 40
n_out = 512
# 训练与测试
eval_interval = 10
num_epoch = 500
margin = 0.2
scale = 30
num_speaker = 5994
lr = 0.001
gamma = 0.95
num_eval = 10
# 模型训练
device = "cuda:2"
log_dir = "/workspace/rwang/competition/voxsrc2020/logs/exp-22"

__all__ = ["load_train", "load_trial", "loadWAV",
           "SpeakerNet", "ResNetSE34L", "AMSoftmax"]


############
#  数据准备 #
############

class waveform(Dataset):
    """Dataset for load utterances among trials."""

    def __init__(self, dataset, load_wav):
        self.dataset = dataset
        self.load_wav = load_wav

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file = self.dataset[idx]
        wave = self.load_wav(file)
        return {"file": file, "wave": wave}


class voxceleb2(Dataset):
    """Sample format: speaker, wav_path
    Example:
    id00012 id00012/21Uxsk56VDQ/00001.wav
    """

    def __init__(self, datalst: np.ndarray, speaker: dict, load_wav):
        super(voxceleb2, self).__init__()
        self.datalst = datalst
        self.speaker = speaker
        self.load_wav = load_wav

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        spk, path = self.datalst[idx]
        label = self.speaker[spk]
        wav = self.load_wav(path)
        wav = torch.FloatTensor(wav)
        return {"label": label, "input": wav}


class vox2trials(Dataset):
    """Trial format: label enroll test
    Example:
    1 voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
    """

    def __init__(self, trials: np.ndarray):
        super(vox2trials, self).__init__()
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        label, enroll, test = self.trials[idx]
        return {"label": label, "enroll": enroll, "test": test}


class voxsampler(Sampler):
    """Indices Resampler for dataset.
    """

    def __init__(self, data_source: pd.DataFrame, speaker_id: dict,
                 max_utt_per_spk: int = 100, batch_size: int = 128):
        self.data_source = data_source.values
        self.speaker_id = speaker_id
        self.data_dict = data_source.groupby(by="speaker").groups.items()
        self.max_utt_per_spk = max_utt_per_spk
        self.batch_size = batch_size

    def __iter__(self):
        flattened_list = []
        # Data for each class
        for speaker, indices in self.data_dict:
            label = self.speaker_id[speaker]
            numUtt = min(len(indices), self.max_utt_per_spk)
            rp = np.random.permutation(indices)[:numUtt]
            # it may be further accelarated.
            flattened_list.extend(rp.tolist())
        # Data in random order
        mixid = np.random.permutation(len(flattened_list))
        mixlabel = []
        mixmap = []
        # Prevent two pairs of the same speaker in the same batch
        for i in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            idx = flattened_list[i]
            speaker = self.speaker_id[self.data_source[idx][0]]
            if speaker not in mixlabel[startbatch:]:
                mixlabel.append(speaker)
                mixmap.append(idx)
        # Iteration size
        self.num_file = len(mixmap)
        self.files = mixmap
        return iter(mixmap)

    def __len__(self):
        return len(self.data_source)


def loadWAV(filename, L=32240, evalmode=True, num_eval=10):
    audio, sr = sf.read(filename, dtype='int16')
    assert sr == 16000, "sample rate is {} != 16000".format(sr)
    audiosize = audio.shape[0]
    if audiosize <= L:
        shortage = math.floor((L - audiosize + 1) / 2)
        audio = np.pad(audio, (shortage, shortage),
                       'constant', constant_values=0)
        audiosize = audio.shape[0]
    if evalmode:
        startframe = np.linspace(0, audiosize-L, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-L))])
    wave = []
    for asf in startframe:
        wave.append(audio[int(asf):int(asf)+L])
    if evalmode is False:
        wave = wave[0]
    else:
        wave = np.stack(wave, axis=0)
    wave = torch.FloatTensor(wave)
    return wave


def load_train(trainlst, traindir, maptrain5994, L=L,
               batch_size=batch_size, num_worker=num_worker,
               max_utt_per_spk=max_utt_per_spk):
    def load_train_wav(path): return loadWAV(path, L=L, evalmode=False)
    df_train = pd.read_csv(trainlst, sep=" ", header=None,
                           names=["speaker", "file"])
    df_train["file"] = df_train["file"].apply(lambda x: traindir + x)
    map_train = dict(pd.read_csv(maptrain5994, header=None).values)
    data = voxceleb2(df_train.values, map_train, load_train_wav)
    sampler = voxsampler(df_train, map_train,
                         max_utt_per_spk=max_utt_per_spk, batch_size=batch_size)
    dataloader = DataLoader(data, batch_size=batch_size,
                            num_workers=num_worker, shuffle=False,
                            sampler=sampler)
    return dataloader


def load_trial(testlst, testdir,
               batch_size=batch_size, num_worker=num_worker):
    df_trials = pd.read_csv(testlst, sep=" ", header=None, names=[
                            "label", "enrollment", "test"])
    df_trials["enrollment"] = df_trials["enrollment"].apply(
        lambda x: testdir + x)
    df_trials["test"] = df_trials["test"].apply(lambda x: testdir + x)
    trials_set = vox2trials(df_trials.values)
    trials_loader = DataLoader(trials_set, batch_size=batch_size,
                               num_workers=num_worker, shuffle=False)
    return trials_loader

##########
# 模型准备
##########


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', **kwargs):

        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        super(ResNetSE, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(
            block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(
            block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(
            block, num_filters[3], layers[3], stride=(1, 1))

        self.avgpool = nn.AvgPool2d((5, 1), stride=1)

        self.instancenorm = nn.InstanceNorm1d(40)
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                            n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(
                num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(
                num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        """
        x (tensor) : B * T
        """
        x = self.torchfb(x) + 1e-6  # B * F * T
        x = self.instancenorm(x.log()).unsqueeze(1).detach()  # B * 1 * F * T

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        if self.encoder_type == "SAP":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * T * C
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def ResNetSE34L(nOut=256, num_filters=[16, 32, 64, 128], **kwargs):
    # Number of filters
    layers = [3, 4, 6, 3]
    model = ResNetSE(SEBasicBlock, layers, num_filters, nOut, **kwargs)
    return model


##########
# 训练与测试
##########

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AMSoftmax(nn.Module):
    def __init__(self, in_feats, n_classes=10, m=0.2, s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(
            in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialized AMSoftmax margin = %.3f scale = %.3f' %
              (self.m, self.s))

    def forward(self, x, label=None):
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)

        if not self.training:
            return F.softmax(costh, dim=1)

        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.to(x.device)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        prec1, prec5 = accuracy(costh_m_s.detach().cpu(),
                                label.detach().cpu(), topk=(1, 5))
        return loss, prec1, prec5


def calculate_score(enroll_feat, test_feat, num_eval=10, mode="norm2"):
    if mode is "norm2":
        dist = F.pairwise_distance(
            enroll_feat.unsqueeze(-1).expand(-1, -1, num_eval),
            test_feat.unsqueeze(-1).expand(-1, -1, num_eval).transpose(0, 2)).detach().cpu().numpy()
        score = -np.mean(dist)
    elif mode is "cosine":
        dist = F.cosine_similarity(
            enroll_feat.unsqueeze(-1).expand(-1, -1, num_eval),
            test_feat.unsqueeze(-1).expand(-1, -1, num_eval).transpose(0, 2)).detach().cpu().numpy()
        score = np.mean(dist)
    return score


def calculate_eer(y, y_score, pos):
    """Calculate Equal Error Rate (EER).

    Parameters
    ----------
    y : array_like, ndim = 1
        y denotes groundtruth scores {0, 1}.
    y_score : array_like, ndim = 1
        y_score denotes the prediction scores.

    Returns
    -------
    eer : array
        EER between y and y_score.
    thresh : float
        threshold of EER.
    """
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, np.float(thresh)

#########
# 模型训练
#########


class SpeakerNet(nn.Module):
    def __init__(self,
                 device=device,
                 log_dir=log_dir,
                 eval_interval=eval_interval,
                 lr=lr,
                 gamma=gamma,
                 top=None,
                 net=None):
        super(SpeakerNet, self).__init__()
        self.net = nn.Sequential(net, top).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.lr_step = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=gamma)
        self.TBoard = SummaryWriter(log_dir=log_dir)
        self.device = device
        self.eval_interval = eval_interval
        self.log_dir = log_dir

    def train(self, dataloader, num_epoch=num_epoch, step_num=0):
        modelst = []
        for epoch in range(num_epoch):
            self.net.train()
            for i_step, batch in tqdm(enumerate(dataloader), total=dataloader.sampler.num_file//dataloader.batch_size):
                self.optimizer.zero_grad()
                label = batch["label"].to(self.device)
                inp = batch["input"].to(self.device)
                loss, prec1, prec5 = self.net[1](self.net[0](inp), label)
                loss.backward()
                self.optimizer.step()
                # TBoard
                step_num += 1
                self.TBoard.add_scalar(
                    'Metrics/train_loss', loss.item(), step_num)
                self.TBoard.add_scalar(
                    'Metrics/lr', self.lr_step.get_last_lr()[0], step_num)
                self.TBoard.add_scalar('Metrics/top1_acc', prec1, step_num)
                self.TBoard.add_scalar('Metrics/top5_acc', prec5, step_num)

            if (epoch + 1) % self.eval_interval == 0:
                self.net.eval()
                modeldir = "snapshot-epoch-%03d.model" % (epoch + 1)
                modeldir = os.path.join(self.log_dir, modeldir)
                torch.save(self.net.state_dict(), modeldir)
                self.lr_step.step()
                modelst.append(modeldir)
        return modelst, step_num, loss.item(), prec1, prec5

    def _prefeat(self, trials_loader, L, num_eval, batch_size, embed_norm):
        def load_trial_wav(path): return loadWAV(
            path, L=L, evalmode=True, num_eval=num_eval)
        wav_file = [[trials_loader.dataset[i]["enroll"],
                     trials_loader.dataset[i]["test"]]
                    for i in range(len(trials_loader.dataset))]
        wav_file = sorted(list(set(np.concatenate(wav_file).tolist())))
        trials = waveform(wav_file, load_trial_wav)

        def collate_fn(batch):
            """collect from a batch of VoxWave Dataset."""
            file = [item["file"] for item in batch]
            wave = torch.cat([item["wave"] for item in batch], dim=0)
            return {"file": file, "wave": wave}

        trialoader = DataLoader(trials, batch_size=batch_size,
                                num_workers=5, collate_fn=collate_fn, shuffle=False)
        self.net.eval()
        trials_feat = {}
        for data in tqdm(trialoader, total=len(trialoader)):
            file = data["file"]
            wave = data["wave"].to(self.device)
            feat = self.net[0](wave)
            if embed_norm is True:
                feat = F.normalize(feat, p=2, dim=1).detach().cpu()
            for i, j in enumerate(range(0, feat.shape[0], num_eval)):
                trials_feat[file[i]] = feat[j: j + num_eval].clone()
        return trials_feat

    def eval(self, trials_loader, step_num, mode="norm2",
             L=eval_L, num_eval=num_eval,
             batch_size=30, embed_norm=True, trials_feat=None):

        if trials_feat is None:
            trials_feat = self._prefeat(trials_loader, L, num_eval,
                                        batch_size, embed_norm)
        all_scores = []
        all_labels = []
        all_trials = []
        for trial in tqdm(trials_loader, total=len(trials_loader)):
            label = trial["label"]
            enroll = trial["enroll"]
            test = trial["test"]
            for i in range(len(label)):
                enroll_embed = trials_feat[enroll[i]]
                test_embed = trials_feat[test[i]]
                score = calculate_score(enroll_embed.to(device),
                                        test_embed.to(device),
                                        mode=mode)
                all_scores.append(score)
                all_trials.append([enroll[i], test[i]])
            all_labels.extend(label.numpy().tolist())
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        eer, thresh = calculate_eer(all_labels, all_scores, 1)
        self.TBoard.add_scalar('Metrics/EER', eer, step_num)
        return eer, thresh, all_scores, all_labels, all_trials, trials_feat

    def load(self, modeldir, strict=True):
        loaded_state = torch.load(modeldir)
        if strict is True:
            self.net.load_state_dict(loaded_state)
        else:
            self_state = self.net.state_dict()
            for name, param in loaded_state.items():
                origname = name
                if name not in self_state:
                    name = name.replace("module.", "")
                    if name not in self_state:
                        print("%s is not in the model." % origname)
                        continue
                if self_state[name].size() != loaded_state[origname].size():
                    print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                        origname, self_state[name].size(), loaded_state[origname].size()))
                    continue
                self_state[name].copy_(param)


if __name__ == "__main__":
    trainlst = "/workspace/rwang/voxceleb/train_list.txt"
    testlst = "/workspace/rwang/VoxSRC2020/data/verif/trials.txt"
    traindir = "/workspace/rwang/voxceleb/voxceleb2/"
    testdir = "/workspace/rwang/voxceleb/"
    maptrain5994 = "/workspace/rwang/competition/voxsrc2020/maptrain5994.txt"
    train = load_train(trainlst=trainlst, traindir=traindir,
                       maptrain5994=maptrain5994)
    trial = load_trial(testlst=testlst, testdir=testdir)
    net = ResNetSE34L(nOut=512, num_filters=[16, 32, 64, 128])
    top = AMSoftmax(in_feats=512, n_classes=5994, m=0.2, s=30)
    snet = SpeakerNet(net=net, top=top)
    modelst, step_num, loss, prec1, prec5 = snet.train(train, num_epoch=1)
    eer, thresh, all_scores, all_labels, all_trials, trials_feat = snet.eval(
        trial, step_num=0, trials_feat=None)
