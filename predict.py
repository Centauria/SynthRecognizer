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
    parser.add_argument('-t', '--dataset', type=str, required=False)
    parser.add_argument('-d', '--dataset-dir', type=str, required=True, help='only to describe output form')
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--save-feature', action='store_true')
    parser.add_argument('files', type=str, nargs='+')
    args = parser.parse_args()

    if args.config is not None:
        conf = config.Config(args.config, **vars(args))
    else:
        if args.model is not None:
            conf = config.Config(**vars(args))
            conf.model = args.model
        else:
            raise ValueError('argument `config` and `model` cannot be None simultaneously')
    print(conf)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    saved = torch.load(args.model_path)
    model = config.get_model(conf).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    synth_set = config.get_dataset(conf)(args.dataset_dir)
    keys = synth_set.info.keys()

    os.makedirs(args.output, exist_ok=True)

    if args.save_feature:
        feature_dir = os.path.join(args.output, 'feature')
        os.makedirs(feature_dir, exist_ok=True)

    with open(os.path.join(args.output, 'result.csv'), 'w') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for file_or_folder in args.files:
            if os.path.isfile(file_or_folder):
                print(f'Processing {file_or_folder}')
                x = synth_set.get(file_or_folder)
                if args.save_feature:
                    feature = model.conv_1d(x.unsqueeze(0).to(device))
                    y = model.conv_2d(feature)
                else:
                    y = model(x.unsqueeze(0).to(device))
                r = synth_set.retrieve(y)
                r['wav_path'] = file_or_folder
            elif os.path.isdir(file_or_folder):
                for file in os.listdir(file_or_folder):
                    file = os.path.join(file_or_folder, file)
                    print(f'Processing {file}')
                    x = synth_set.get(file)
                    if isinstance(x, tuple):
                        x = list(map(lambda c: c.unsqueeze(0).to(device), x))
                    else:
                        x = x.unsqueeze(0).to(device)
                    if args.save_feature:
                        feature = model.conv_1d(x)
                        y = model.conv_2d(feature)
                    else:
                        y = model(x)
                    r = synth_set.retrieve(y)
                    r['wav_path'] = file
            else:
                raise ValueError('this is not gonna happen')
            writer.writerow(r)
