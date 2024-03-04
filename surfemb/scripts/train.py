from pathlib import Path
import argparse

import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import random
import numpy as np

from .. import utils
from ..data import obj, instance
from ..data.config import config
from ..surface_embedding import SurfaceEmbeddingModel


def worker_init_fn(*_):
    # each worker should only use one os thread
    # numpy/cv2 takes advantage of multithreading by default
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import cv2
    cv2.setNumThreads(0)

    # random seed
    import numpy as np
    np.random.seed(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--subsets', type=str, nargs='*', default=None)
    parser.add_argument('--subset-fractions', type=float, nargs='*', default=None,
                        help="Defines per-subset fractions (0.0, 1.0] to use for training, defined in the same order "
                             "as the subsets and starting from the left. Defaults to 1.0 for each subset.")
    parser.add_argument('--subset-balance', choices=['concat', 'balanced'], default='concat')
    parser.add_argument('--n-valid', type=float, default=200,
                        help="Any value 0.0 < n-valid < 1 will be interpreted relative to the subset size, A value > 1 "
                             "will be interpreted as the total number (evenly sampled from all subsets).")
    parser.add_argument('--res-data', type=int, default=256)
    parser.add_argument('--res-crop', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--min-visib-fract', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=500_000)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--objs', type=int, nargs='*', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--lr-range-test', action='store_true')

    parser = SurfaceEmbeddingModel.model_specific_args(parser)
    args = parser.parse_args()

    assert 0.0 < args.n_valid != 1.0, f"Invalid argument for n_valid: {args.n_valid}"

    debug = args.debug
    root = Path('data/bop') / args.dataset

    # load configurations for all subsets, or use default one
    if args.subsets is None:
        cfgs = {args.dataset: config[args.dataset]}
    else:
        cfgs = {}
        for subset in args.subsets:
            subset_id = f"{args.dataset}_{subset}" if subset != "default" else args.dataset
            assert subset_id in config, f"Unknown subset for {args.dataset}: {subset}"
            assert subset_id not in cfgs, f"Duplicate subset: {subset_id}"
            cfgs[subset_id] = config[subset_id]

    # check subset fractions
    if args.subset_fractions is not None:
        for fraction in args.subset_fractions:
            assert 0.0 < fraction <= 1.0, f"Invalid subset fraction: {fraction}"

    # load objs
    objs, obj_ids = obj.load_objs(root / list(cfgs.values())[0].model_folder, args.objs)
    assert len(obj_ids) > 0

    # model
    if args.ckpt:
        assert args.dataset == Path(args.ckpt).name.split('-')[0]
        assert Path(args.ckpt).exists(), f"FileNotFound: {args.ckpt}"
        model = SurfaceEmbeddingModel.load_from_checkpoint(args.ckpt)
    else:
        model = SurfaceEmbeddingModel(n_objs=len(obj_ids), **vars(args))

    # datasets
    auxs = model.get_auxs(objs, args.res_crop)
    data_subsets = {}
    for idx, (subset_id, cfg) in enumerate(cfgs.items()):
        subset = instance.BopInstanceDataset(
            dataset_root=root, test=False, configs=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=args.min_visib_fract, scene_ids=[1] if debug else None,
        )
        assert len(subset) > 0, f"Loaded empty dataset: {subset_id}"
        data_subsets[subset_id] = subset

    n_subsets = len(data_subsets)
    n_valid = args.n_valid
    if args.n_valid < 1.0:
        n_valid_per_subset = np.ceil(n_valid / n_subsets)
    train_subsets = []
    val_data = utils.EmptyDataset()

    # Train-validation split
    for idx, (subset_id, subset) in enumerate(data_subsets.items()):
        # split validation data
        if args.n_valid < 1.0:
            n_valid_subset = n_valid_per_subset
        else:
            n_valid_subset = np.ceil(len(subset) * args.n_valid)
        train_subset, val_subset = torch.utils.data.random_split(
            subset, (len(subset) - n_valid_subset, n_valid_subset),
            generator=torch.Generator().manual_seed(0),
        )
        val_data += val_subset
        # optional: discard fraction of training data
        if args.subset_fractions is not None and idx < len(args.subset_fractions):
            subset_frac = args.subset_fractions[idx]
            keep_count = np.ceil(subset_frac * len(train_subset))
            print(f"Info: Keeping {keep_count} ({subset_frac:02.2%}) for subset {subset_id}.")
            random.shuffle(train_subset.indices)
            train_subset.indices = train_subset.indices[:keep_count]
        train_subsets.append(train_subset)

    # Create training dataset
    train_data = utils.EmptyDataset()
    n_train_samples = sum([len(d) for d in train_subsets])
    if args.subset_balance == "concat":
        for subset in train_subsets:
            train_data += subset
    elif args.subset_balance == "balanced":
        train_data = utils.balanced_dataset_concat(train_subsets)
        print(f"Info: Balanced dataset changed training dataset size from {n_train_samples} to {len(train_data)}.")

    print(f"Info: Training samples: {len(train_data)}")
    print(f"Info: Validation samples: {len(val_data)}")

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=torch.get_num_threads() if args.num_workers is None else args.num_workers,
        persistent_workers=True, shuffle=True,
        worker_init_fn=worker_init_fn, pin_memory=True,
    )
    loader_train = torch.utils.data.DataLoader(train_data, drop_last=True, **loader_args)
    # don't shuffle validation set
    loader_args["shuffle"] = False
    loader_valid = torch.utils.data.DataLoader(val_data, **loader_args)

    # train
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project='surfemb', dir=log_dir)
    run.name = run.id
    run.config["samples_train"] = len(train_data)
    run.config["samples_val"] = len(val_data)

    logger = pl.loggers.WandbLogger(experiment=run)
    logger.log_hyperparams(args)

    ckpt_filename = f"{args.dataset}-{run.id}-" + "{epoch}-{step}-{val_loss:.2f}"
    model_ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='data/models/', filename=ckpt_filename, monitor="valid/loss", save_top_k=1, save_last=True)
    model_ckpt_cb.CHECKPOINT_NAME_LAST = f"{args.dataset}-{run.id}-last"

    trainer = pl.Trainer(
        logger=logger, accelerator='gpu', devices=args.gpus, max_steps=args.max_steps,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            model_ckpt_cb,
        ],
        val_check_interval=min(1., n_valid / (len(train_data) + len(val_data)) * 50),  # spend ~1/50th of the time on validation
        auto_lr_find=args.lr_range_test
    )

    if args.lr_range_test:
        print("LR Range Test:")
        #trainer.tune(model, loader_train, loader_valid)
        lr_finder = trainer.tuner.lr_find(model, loader_train, loader_valid, num_training=1000)
        fig = lr_finder.plot(suggest=True)
        fig.savefig(f'data/logs/{args.dataset}-{run.id}-lr-range-test.png')
        model.lr = lr_finder.suggestion()
        print(f"Estimated learning rate is {model.lr:.6f}.")
    else:
        trainer.fit(model, loader_train, loader_valid, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main()
