# -*- coding: utf-8 -*-
import importlib
import json
from typing import Optional

import yaml


class Object(dict):
    """Makes a dict behave like an object, with attribute-style access."""
    
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return None
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __copy__(self):
        return Object(super(Object, self))


class Config(Object):
    def __init__(self, yaml_path: Optional[str] = None, **kwargs):
        super(Config, self).__init__()
        if yaml_path is not None:
            with open(yaml_path) as f:
                c = yaml.load(f, yaml.SafeLoader)
            for k, v in c.items():
                if isinstance(v, dict):
                    self[k] = Config(**v)
                elif isinstance(v, Object):
                    self[k] = Config(**v)
                elif isinstance(v, Config):
                    self[k] = v
                else:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v
    
    def __repr__(self):
        return json.dumps(self, indent=2)
    
    def __copy__(self):
        return Config(**super(Config, self))
    
    def __setitem__(self, key, value):
        keys = key.split('.')
        d = self
        for i in range(len(keys) - 1):
            try:
                d = super(Config, d).__getitem__(keys[i])
            except KeyError:
                super(Config, d).__setitem__(keys[i], Config())
                d = super(Config, d).__getitem__(keys[i])
        else:
            try:
                original = super(Config, d).__getitem__(keys[-1])
                super(Config, d).__setitem__(keys[-1], type(original)(value))
            except (KeyError, ValueError):
                parsed_value = None
                if isinstance(value, Config):
                    parsed_value = value
                elif isinstance(value, str):
                    try:
                        parsed_value = int(value)
                    except (ValueError, TypeError):
                        pass
                    if parsed_value is None:
                        try:
                            parsed_value = float(value)
                        except (ValueError, TypeError):
                            pass
                    if parsed_value is None:
                        parsed_value = value
                else:
                    parsed_value = value
                super(Config, d).__setitem__(keys[-1], parsed_value)


def get_model(conf: dict):
    module = importlib.import_module(f'models.{conf["model"]}')
    if hasattr(module, 'model'):
        return module.model
    elif hasattr(module, 'get_model'):
        return module.get_model(conf)
    else:
        raise NotImplementedError('submodules of `model` must implement `model` or `get_model()`')


def get_optimizer(model, conf: dict):
    kwargs = conf['optimizer'].copy()
    del kwargs['name']
    return vars(importlib.import_module('torch.optim'))[conf['optimizer']['name']](
        model.parameters(), **kwargs)


def get_dataset(conf: dict):
    return vars(importlib.import_module('data.dataset'))[conf['dataset']]


def get_criterion(conf: Config):
    v = vars(importlib.import_module('engine.loss'))
    criterions = {k: v[c]() for k, c in conf['criterion'].items()}
    loss_weights = conf.loss_weight
    
    def criterion(outputs, targets, weights=loss_weights):
        if weights is None:
            weights = {k: 1.0 for k in criterions.keys()}
        loss = sum([
            weights[w] * criterions[c](outputs[o], targets[t])
            for w, c, o, t in zip(weights, criterions, outputs, targets)
        ])
        return loss
    
    return criterion


def get_metrics(conf: dict):
    v = vars(importlib.import_module('engine.metrics'))
    metrics = {}
    for k, ms in conf['metrics'].items():
        for m in ms:
            metrics[f'{k}_{m}'] = v[m](lambda x, k=k: (x[0][k], x[1][k]))
            # Have to freeze `i` in the lambda expression
            # Alternative method: (lambda k: lambda x: (x[0][k], x[1][k]))(k)
    return metrics
