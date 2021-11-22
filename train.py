# -*- coding: utf-8 -*-
import argparse

import torch
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.metrics import Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import data

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train', allow_abbrev=False)
    parser.add_argument('name')
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('-k', '--checkpoint', type=str, required=False)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--checkpoint-dir', type=str)
    parser.add_argument('-d', '--dataset-dir', required=True)
    _, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith(('-', '--')):
            parser.add_argument(arg)

    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = f'log/{args.name}'
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'checkpoints/{args.name}'

    if args.config is not None:
        conf = config.Config(args.config, **vars(args))
    else:
        conf = config.Config(**vars(args))
    print(conf)

    logger = SummaryWriter(log_dir=conf.log_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    synth_set = config.get_dataset(conf)(conf.dataset_dir)
    dset_train, dset_eval = data.random_split(synth_set, [11, 1])
    loader_train = DataLoader(dset_train, conf.batchsize, shuffle=True, num_workers=8)
    loader_eval = DataLoader(dset_eval, conf.batchsize, shuffle=True, num_workers=8)

    criterion = config.get_criterion(conf)
    model = config.get_model(conf).to(device)
    print(f'Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    optimizer = config.get_optimizer(model, conf)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=5, cooldown=5, min_lr=1e-6)
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    if conf.checkpoint is not None:
        checkpoint = torch.load(conf.checkpoint)
        trainer.load_state_dict(checkpoint['trainer'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    val_metrics = config.get_metrics(conf)
    val_metrics['criterion'] = Loss(criterion)
    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device,
        output_transform=lambda x, y, y_pred: (y_pred, y)
    )


    @trainer.on(Events.ITERATION_COMPLETED(every=5))
    def log_training_loss(e: Engine):
        print(f"Epoch[{e.state.epoch}] iter[{e.state.iteration}] Loss: {e.state.output}")
        logger.add_scalar("Train: loss", e.state.output, global_step=e.state.iteration)


    @trainer.on(Events.EPOCH_STARTED)
    def log_training_info(e: Engine):
        lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch[{e.state.epoch}] lr={lr}')
        logger.add_scalar('lr', lr, e.state.epoch)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(e: Engine):
        evaluator.run(loader_train)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - "
            f"Epoch: {e.state.epoch}  "
            f"criterion: {metrics['criterion']:.6f}  "
            f"y_args_rmse: {metrics['y_args_rmse']:.6f}  "
            f"y_args_lcl: {metrics['y_args_lcl']:.6f}  "
            f"y_args_mae: {metrics['y_args_mae']:.6f}  "
            f"y_kind_ce: {metrics['y_kind_ce']:.6f}"
        )
        logger.add_scalars('Eval: train set', metrics, e.state.epoch)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(e: Engine):
        evaluator.run(loader_eval)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - "
            f"Epoch: {e.state.epoch}  "
            f"criterion: {metrics['criterion']:.6f}  "
            f"y_args_rmse: {metrics['y_args_rmse']:.6f}  "
            f"y_args_lcl: {metrics['y_args_lcl']:.6f}  "
            f"y_args_mae: {metrics['y_args_mae']:.6f}  "
            f"y_kind_ce: {metrics['y_kind_ce']:.6f}"
        )
        logger.add_scalars('Eval: val set', metrics, e.state.epoch)


    @evaluator.on(Events.COMPLETED)
    def lr_scheduler_step(e: Engine):
        lr_scheduler.step(e.state.metrics['criterion'])


    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        Checkpoint({
            'trainer': trainer,
            'model': model,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }, DiskSaver(conf.checkpoint_dir, create_dir=True, require_empty=False))
    )
    evaluator.add_event_handler(
        Events.COMPLETED,
        EarlyStopping(10, lambda engine: -engine.state.metrics['criterion'], trainer)
    )

    trainer.run(loader_train, max_epochs=500)
