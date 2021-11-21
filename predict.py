# -*- coding: utf-8 -*-
import argparse

import torch
import torchaudio

import models.cnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('files', type=str, nargs='+')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    saved = torch.load(args.model)
    model = models.cnn.CNN.to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    
    for file in args.files:
        w, sr = torchaudio.load(file)
        w = torch.mean(w, 0, keepdim=True)
        trans = torchaudio.transforms.MFCC(sr, 30, log_mels=True)
        mfcc = trans(w)
        mfcc = mfcc[:, :, :64]
        y = model(mfcc.unsqueeze(0).to(device))
        print(f'Result for "{file}": '
              f'{y.detach().cpu().numpy()}')
