import json
from pathlib import Path
from typing import Sequence, Union
import warnings

import numpy as np
from tqdm import tqdm
import torch.utils.data

from .config import DatasetConfig


# BopInstanceDataset should only be used with test=True for debugging reasons
# use detector_crops.DetectorCropDataset for actual test inference


class BopInstanceDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_root: Path, test: bool, configs: Union[DatasetConfig, Sequence[DatasetConfig]],
            obj_ids: Sequence[int], scene_ids=None, targets=None, min_visib_fract=0.1, min_px_count_visib=1024,
            auxs: Sequence['BopInstanceAux'] = tuple(), show_progressbar=True,
    ):
        assert dataset_root.exists(), f"Invalid dataset directory: {dataset_root}"
        assert scene_ids is None or targets is None, "Can't have both yet."
        assert targets is None or test, "Targets can only be used for test split"

        self.test = test
        self.dataset_root = dataset_root
        self.configs = configs if isinstance(configs, Sequence) else [configs]

        assert not test or len(self.configs) == 1, "Only one test subset at a time supported yet."

        self.auxs = auxs
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.instances = []
        for cfg in self.configs:
            subset_name = cfg.test_folder if test else cfg.train_folder
            data_folder = dataset_root / subset_name
            if scene_ids is None:
                scene_ids = sorted([int(p.name) for p in data_folder.glob('*') if not p.is_file()])
            if targets is not None:
                scene_ids = [int(s) for s in scene_ids if s in targets]
            for scene_id in tqdm(scene_ids, 'loading crop info') if show_progressbar else scene_ids:
                scene_folder = data_folder / f'{scene_id:06d}'
                assert (scene_folder / 'scene_gt.json').exists(), f"FileNotFound: {(scene_folder / 'scene_gt.json')}"
                assert (scene_folder / 'scene_gt_info.json').exists(), f"FileNotFound: {(scene_folder / 'scene_gt_info.json')}"
                assert (scene_folder / 'scene_camera.json').exists(), f"FileNotFound: {(scene_folder / 'scene_camera.json')}"
                scene_gt = json.load((scene_folder / 'scene_gt.json').open())
                scene_gt_info = json.load((scene_folder / 'scene_gt_info.json').open())
                scene_camera = json.load((scene_folder / 'scene_camera.json').open())

                for img_id, poses in scene_gt.items():
                    if targets is not None and int(img_id) not in targets[scene_id]:
                        continue
                    img_info = scene_gt_info[img_id]
                    K = np.array(scene_camera[img_id]['cam_K']).reshape((3, 3)).copy()

                    for pose_idx, pose in enumerate(poses):
                        obj_id = pose['obj_id']
                        if targets is not None and obj_id not in targets[scene_id][int(img_id)]:
                            continue
                        if obj_ids is not None and obj_id not in obj_ids:
                            continue
                        pose_info = img_info[pose_idx]
                        if pose_info['visib_fract'] < min_visib_fract:
                            continue
                        if pose_info['px_count_visib'] < min_px_count_visib:
                            continue

                        bbox_visib = pose_info['bbox_visib']
                        bbox_obj = pose_info['bbox_obj']

                        cam_R_obj = np.array(pose['cam_R_m2c']).reshape(3, 3)
                        cam_t_obj = np.array(pose['cam_t_m2c']).reshape(3, 1)

                        rgb_path = data_folder / f'{scene_id:06d}/{cfg.img_folder}/{int(img_id):06d}.{cfg.img_ext}'
                        depth_path = data_folder / f'{scene_id:06d}/{cfg.depth_folder}/{int(img_id):06d}.{cfg.depth_ext}'
                        mask_path = data_folder / f'{scene_id:06d}/mask/{int(img_id):06d}_{pose_idx:06d}.{cfg.mask_ext}'
                        mask_visib_path = data_folder / f'{scene_id:06d}/mask_visib/{int(img_id):06d}_{pose_idx:06d}.{cfg.mask_ext}'

                        self.instances.append(dict(
                            scene_id=scene_id, img_id=int(img_id), K=K, obj_id=obj_id, pose_idx=pose_idx,
                            bbox_visib=bbox_visib, bbox_obj=bbox_obj, cam_R_obj=cam_R_obj, cam_t_obj=cam_t_obj,
                            obj_idx=obj_idxs[obj_id], rgb_path=rgb_path, depth_path=depth_path, mask_path=mask_path,
                            mask_visib_path=mask_visib_path,
                        ))

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        try:
            for aux in self.auxs:
                instance = aux(instance, self)
        except Exception as e:
            print(f"Caught {type(e).__name__} when loading sample {i}: \n{e}")
            instance = None
        return instance


class BopInstanceAux:
    def init(self, dataset: BopInstanceDataset):
        pass

    def __call__(self, data: dict, dataset: BopInstanceDataset) -> dict:
        pass
