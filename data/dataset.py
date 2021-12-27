# -*- coding: utf-8 -*-
import os.path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, AmplitudeToDB


class SynthSet(Dataset):
    def __init__(self, dataset_folder=None):
        self.sr = 44100
        if dataset_folder is not None:
            self.info_path = os.path.join(dataset_folder, 'info.csv')
            self.info = pd.read_csv(self.info_path)
            self.wav_dir = os.path.join(dataset_folder, 'wav')

    def __len__(self):
        return len(self.info)

    def get(self, wav_path):
        x, sr = torchaudio.load(wav_path)
        assert sr == self.sr
        return x

    def __getitem__(self, item):
        values = self.info.loc[item]
        wav_path = os.path.join(self.wav_dir, values['wav_path'])
        x = self.get(wav_path)
        y = torch.tensor(values.values[:4].astype(np.float32))
        return x, y


class SynthSetLPS(SynthSet):
    def __init__(self, dataset_folder=None):
        super(SynthSetLPS, self).__init__(dataset_folder)
        self.spec = Spectrogram(512, hop_length=256)
        self.a_d = AmplitudeToDB()

    def get(self, wav_path):
        x = super(SynthSetLPS, self).get(wav_path)
        x = self.spec(x.mean(dim=0, keepdim=True))
        x = self.a_d(x)
        return x

    def __getitem__(self, item):
        x, y = super(SynthSetLPS, self).__getitem__(item)
        x = self.spec(x.mean(dim=0, keepdim=True))
        x = self.a_d(x)
        return x[:, :, :64].transpose(1, 2), y


class SynthSetHybrid(SynthSetLPS):
    def get(self, wav_path):
        x = super(SynthSetLPS, self).get(wav_path)
        s = self.spec(x.mean(dim=0, keepdim=True))
        s = self.a_d(s)
        return x, s

    def __getitem__(self, item):
        x, y = super(SynthSetLPS, self).__getitem__(item)
        s = self.spec(x.mean(dim=0, keepdim=True))
        s = self.a_d(s)
        return (x, s[:, :, :64].transpose(1, 2)), y


class SynthSetClassify(SynthSet):
    def __getitem__(self, item):
        values = self.info.loc[item]
        wav_path = os.path.join(self.wav_dir, values['wav_path'])
        x = self.get(wav_path)
        y_args = np.hstack((
            values.values[:5].astype(np.float32),
            values.values[6:11].astype(np.float32)
        ))
        y_kind = torch.tensor(values.values[5] - 1, dtype=torch.long)
        y = {
            'y_args': y_args,
            'y_kind': y_kind
        }
        return x, y

    def retrieve(self, result):
        y_args = result['y_args'][0].detach().cpu().numpy().tolist()
        y_kind = torch.argmax(result['y_kind'], dim=-1)[0].detach().cpu().numpy().tolist()
        r = y_args[:5] + [y_kind + 1] + y_args[5:]
        return dict(zip(self.info.keys(), r))


class SynthSetLPSClassify(SynthSetClassify):
    def __init__(self, dataset_folder=None):
        super(SynthSetLPSClassify, self).__init__(dataset_folder)
        self.spec = Spectrogram(512, hop_length=256)
        self.a_d = AmplitudeToDB()

    def get(self, wav_path):
        x = super(SynthSetLPSClassify, self).get(wav_path)
        x = self.spec(x.mean(dim=0, keepdim=True))
        x = self.a_d(x)
        x = x.transpose(1, 2)
        return x

    def __getitem__(self, item):
        x, y = super(SynthSetLPSClassify, self).__getitem__(item)
        x = self.spec(x.mean(dim=0, keepdim=True))
        x = self.a_d(x)
        x = x.transpose(1, 2)
        return x, y


class SynthSetLPSMiniClassify(SynthSetLPSClassify):
    def __getitem__(self, item):
        x, y = super(SynthSetLPSMiniClassify, self).__getitem__(item)
        return x[:, :64, :], y


class SynthSetHybridClassify(SynthSetLPSClassify):
    def get(self, wav_path):
        x = super(SynthSetLPSClassify, self).get(wav_path)
        s = self.spec(x.mean(dim=0, keepdim=True))
        s = self.a_d(s)
        s = s.transpose(1, 2)
        return x, s

    def __getitem__(self, item):
        x, y = super(SynthSetLPSClassify, self).__getitem__(item)
        s = self.spec(x.mean(dim=0, keepdim=True))
        s = self.a_d(s)
        s = s.transpose(1, 2)
        return (x, s), y
