from pathlib import Path
import argparse

import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

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
    parser.add_argument('--n-valid', type=int, default=200)
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
    parser.add_argument('--no-synth', dest='synth', action='store_false')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--lr-range-test', action='store_true')
    parser.add_argument('--train_fraction', type=float, default=1.0)

    parser = SurfaceEmbeddingModel.model_specific_args(parser)
    args = parser.parse_args()
    debug = args.debug
    root = Path('data/bop') / args.dataset
    cfg = config[args.dataset]

    # load objs
    objs, obj_ids = obj.load_objs(root / cfg.model_folder, args.objs)
    assert len(obj_ids) > 0

    # model
    if args.ckpt:
        assert args.dataset == Path(args.ckpt).name.split('-')[0]
        model = SurfaceEmbeddingModel.load_from_checkpoint(args.ckpt)
    else:
        model = SurfaceEmbeddingModel(n_objs=len(obj_ids), **vars(args))

    # datasets
    auxs = model.get_auxs(objs, args.res_crop)
    data = utils.EmptyDataset()
    if args.synth:
        data += instance.BopInstanceDataset(
            dataset_root=root, synth=True, pbr=True, test=False, cfg=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=args.min_visib_fract, scene_ids=[1] if debug else None,
        )
        assert len(data) > 0, "Loaded empty pbr dataset!"
        data += instance.BopInstanceDataset(
            dataset_root=root, synth=True, pbr=False, test=False, cfg=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=args.min_visib_fract, scene_ids=[1] if debug else None,
        )
        assert len(data) > 0, "Loaded empty synthetic dataset!"
    if args.real:
        assert args.dataset in {'tless', 'tudl', 'ycbv', 'mvpsp'}
        data_real = instance.BopInstanceDataset(
            dataset_root=root, synth=False, pbr=False, test=False, cfg=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=args.min_visib_fract, scene_ids=[1] if debug else None, train_fraction=args.train_fraction
        )
        assert len(data_real) > 0, "Loaded empty real dataset!"
        if args.synth:
            data = utils.balanced_dataset_concat(data, data_real)
        else:
            data = data_real

    n_valid = args.n_valid
    data_train, data_valid = torch.utils.data.random_split(
        data, (len(data) - n_valid, n_valid),
        generator=torch.Generator().manual_seed(0),
    )

    print(f"Training samples: {len(data_train)}")
    print(f"Validation samples: {len(data_valid)}")

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=torch.get_num_threads() if args.num_workers is None else args.num_workers,
        persistent_workers=True, shuffle=True,
        worker_init_fn=worker_init_fn, pin_memory=True,
    )
    loader_train = torch.utils.data.DataLoader(data_train, drop_last=True, **loader_args)
    # don't shuffle validation set
    loader_args["shuffle"] = False
    loader_valid = torch.utils.data.DataLoader(data_valid, **loader_args)

    # train
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project='surfemb', dir=log_dir)
    run.name = run.id
    run.config["samples_train"] = len(data_train)
    run.config["samples_val"] = len(data_valid)

    logger = pl.loggers.WandbLogger(experiment=run)
    logger.log_hyperparams(args)

    model_ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='data/models/', save_top_k=1, save_last=True)
    model_ckpt_cb.CHECKPOINT_NAME_LAST = f'{args.dataset}-{run.id}'

    trainer = pl.Trainer(
        resume_from_checkpoint=args.ckpt,
        logger=logger, accelerator='gpu', devices=args.gpus, max_steps=args.max_steps,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            model_ckpt_cb,
        ],
        val_check_interval=min(1., n_valid / len(data) * 50),  # spend ~1/50th of the time on validation
        auto_lr_find=args.lr_range_test
    )

    if args.lr_range_test:
        print("LR Range Test:")
        #trainer.tune(model, loader_train, loader_valid)
        lr_finder = trainer.tuner.lr_find(model, loader_train, loader_valid, min_lr=1e-6, max_lr=1e-1, num_training=200)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        model.lr = lr_finder.suggestion()
        print(f"Estimated learning rate is {model.lr:.6f}.")
    else:
        trainer.fit(model, loader_train, loader_valid)


if __name__ == '__main__':
    main()
