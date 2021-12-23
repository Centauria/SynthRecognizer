# -*- coding: utf-8 -*-
import argparse
import csv
import os.path

import torch

import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('-m', '--model', type=str, required=False)
    parser.add_argument('-p', '--model-path', type=str, required=True)
    parser.add_argument('-d', '--dataset-dir', type=str, required=True, help='only to describe output form')
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('files', type=str, nargs='+')
    args = parser.parse_args()
    
    if args.config is not None:
        conf = config.Config(args.config, **vars(args))
    else:
        conf = config.Config(**vars(args))
    print(conf)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    saved = torch.load(args.model_path)
    model = config.get_model(conf).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    synth_set = config.get_dataset(conf)(args.dataset_dir)
    keys = synth_set.info.keys()
    
    with open(args.output, 'w') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for file_or_folder in args.files:
            if os.path.isfile(file_or_folder):
                print(f'Processing {file_or_folder}')
                x = synth_set.get(file_or_folder)
                y = model(x.unsqueeze(0).to(device))
                r = synth_set.retrieve(y)
                r['wav_path'] = file_or_folder
            elif os.path.isdir(file_or_folder):
                for file in os.listdir(file_or_folder):
                    file = os.path.join(file_or_folder, file)
                    print(f'Processing {file}')
                    x = synth_set.get(file)
                    y = model(x.unsqueeze(0).to(device))
                    r = synth_set.retrieve(y)
                    r['wav_path'] = file
            else:
                raise ValueError('this is not gonna happen')
            writer.writerow(r)
