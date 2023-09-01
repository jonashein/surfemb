import argparse
from pathlib import Path
import json

import numpy as np
import torch
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from .. import utils
from ..data import detector_crops, instance
from ..data.config import config
from ..data.obj import load_objs
from ..data.renderer import ObjCoordRenderer
from ..surface_embedding import SurfaceEmbeddingModel
from ..pose_est import estimate_pose
from ..pose_refine import refine_pose

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--res-data', type=int, default=256)
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--gt-crop', action='store_true')
parser.add_argument('--objs', type=int, nargs='*', default=None)
parser.add_argument('--max-poses', type=int, default=10000)
parser.add_argument('--max-pose-evaluations', type=int, default=1000)
parser.add_argument('--no-rotation-ensemble', dest='rotation_ensemble', action='store_false')
parser.add_argument('--targets-path', type=str, default=None)
args = parser.parse_args()

res_crop = args.res_crop
device = torch.device(args.device)
model_path = Path(args.model_path)
assert model_path.is_file()
model_name = model_path.name.split('.')[0]
dataset = model_name.split('-')[0]

results_dir = Path('data/results')
results_dir.mkdir(exist_ok=True)
poses_fp = results_dir / f'{model_name}-poses.npy'
poses_scores_fp = results_dir / f'{model_name}-poses-scores.npy'
poses_timings_fp = results_dir / f'{model_name}-poses-timings.npy'
scene_ids_fp = results_dir / f'{dataset}-scene_ids.npy'
img_ids_fp = results_dir / f'{dataset}-view_ids.npy'
obj_ids_fp = results_dir / f'{dataset}-obj_ids.npy'
for fp in poses_fp, poses_scores_fp, poses_timings_fp:
    assert not fp.exists()

# load model
model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path)).eval().to(device)  # type: SurfaceEmbeddingModel
model.freeze()

if args.targets_path:
    targets_path = Path(args.targets_path)
    assert targets_path.is_file()
    targets_raw = json.load(targets_path.open("r"))
    targets = {}
    for t in targets_raw:
        scene = int(t["scene_id"])
        im_id = int(t["im_id"])
        obj_id = int(t["obj_id"])
        if scene not in targets:
            targets[scene] = {}
        if im_id not in targets[scene]:
            targets[scene][im_id] = []
        targets[scene][im_id].append(obj_id)

# load data
root = Path('data/bop') / dataset
cfg = config[dataset]
objs, obj_ids = load_objs(root / cfg.model_folder, args.objs)
assert len(obj_ids) > 0
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)

if args.gt_crop:
    auxs = model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=False)
    dataset_args = dict(dataset_root=root, obj_ids=obj_ids, auxs=auxs, cfg=cfg, targets=targets)
    data = instance.BopInstanceDataset(**dataset_args, synth=False, pbr=False, test=True)
else:
    data = detector_crops.DetectorCropDataset(
        dataset_root=root, cfg=cfg, obj_ids=obj_ids,
        detection_folder=Path(f'data/detection_results/{dataset}'),
        auxs=model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=True)
    )

renderer = ObjCoordRenderer(objs, w=res_crop, h=res_crop)

# infer
all_poses = np.empty((2, len(data), 3, 4))
all_scores = np.ones(len(data)) * -np.inf
all_scene_ids = np.empty((len(data)))
all_img_ids = np.empty((len(data)))
all_obj_ids = np.empty((len(data)))
time_forward, time_pnpransac, time_refine = [], [], []


def infer(i, d):
    obj_idx = d['obj_idx']
    img = d['rgb_crop']
    K_crop = d['K_crop']
    all_scene_ids[i] = d['scene_id']
    all_img_ids[i] = d['img_id']
    all_obj_ids[i] = d['obj_id']

    with utils.add_timing_to_list(time_forward):
        mask_lgts, query_img = model.infer_cnn(img, obj_idx, rotation_ensemble=args.rotation_ensemble)
        mask_lgts[0, 0].item()  # synchronize for timing

        # keys are independent of input (could be cached, but it's not the bottleneck)
        obj_ = objs[obj_idx]
        verts = surface_samples[obj_idx]
        verts_norm = (verts - obj_.offset) / obj_.scale
        obj_keys = model.infer_mlp(torch.from_numpy(verts_norm).float().to(device), obj_idx)
        verts = torch.from_numpy(verts).float().to(device)

    with utils.add_timing_to_list(time_pnpransac):
        R_est, t_est, scores, *_ = estimate_pose(
            mask_lgts=mask_lgts, query_img=query_img,
            obj_pts=verts, obj_normals=surface_sample_normals[obj_idx], obj_keys=obj_keys,
            obj_diameter=obj_.diameter, K=K_crop,
        )
        success = len(scores) > 0
        if success:
            best_idx = torch.argmax(scores).item()
            all_scores[i] = scores[best_idx].item()
            R_est, t_est = R_est[best_idx].cpu().numpy(), t_est[best_idx].cpu().numpy()[:, None]
        else:
            R_est, t_est = np.eye(3), np.zeros((3, 1))

    with utils.add_timing_to_list(time_refine):
        if success:
            R_est_r, t_est_r, score_r = refine_pose(
                R=R_est, t=t_est, query_img=query_img, K_crop=K_crop,
                renderer=renderer, obj_idx=obj_idx, obj_=obj_, model=model, keys_verts=obj_keys,
            )
        else:
            R_est_r, t_est_r = R_est, t_est

    for j, (R, t) in enumerate([(R_est, t_est), (R_est_r, t_est_r)]):
        all_poses[j, i, :3, :3] = R
        all_poses[j, i, :3, 3:] = t


for i, d in enumerate(tqdm(BackgroundGenerator(data, max_prefetch=2), desc='running pose est.', smoothing=0, total=len(data))):
    try:
        #print(f"Sample {i}: Scene id {d['scene_id']}, Image id {d['img_id']}, Obj id {d['obj_id']}, Obj idx {d['obj_idx']}")
        infer(i, d)
    except Exception as e:
        print(f"Caught {type(e).__name__} during inference of sample {i}: \n{e}\nSample {i}: {d}")

time_forward = np.array(time_forward)
time_pnpransac = np.array(time_pnpransac)
time_refine = np.array(time_refine)

timings = np.stack((
    time_forward + time_pnpransac,
    time_forward + time_pnpransac + time_refine
))

np.save(str(poses_fp), all_poses)
np.save(str(poses_scores_fp), all_scores)
np.save(str(poses_timings_fp), timings)
np.save(str(scene_ids_fp), all_scene_ids)
np.save(str(img_ids_fp), all_img_ids)
np.save(str(obj_ids_fp), all_obj_ids)
