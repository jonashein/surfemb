from pathlib import Path
import argparse
from collections import defaultdict

import numpy as np

from ...data.config import config

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('poses')
parser.add_argument('--dont-use-refinement', dest='use_refinement', action='store_false')
parser.add_argument('--dont-use-pose-score', dest='use_pose_score', action='store_false')
parser.add_argument('--detection', action='store_true')
args = parser.parse_args()

poses_fp = Path(args.poses)

name = '-'.join(poses_fp.name.split('-')[:-1])  # dataset, run_id, [optionally "depth"]
pose_scores_fp = poses_fp.parent / f'{name.replace("-depth", "")}-poses-scores.npy'
pose_timings_fp = poses_fp.parent / f'{name}-poses-timings.npy'

poses = np.load(str(poses_fp))
pose_scores = np.load(str(pose_scores_fp))
pose_timings = np.load(str(pose_timings_fp))
if args.detection:
    detection_path = Path('data/detection_results') / args.dataset
    det_scene_ids = np.load(str(detection_path / 'scene_ids.npy'))
    det_view_ids = np.load(str(detection_path / 'view_ids.npy'))
    det_obj_ids = np.load(str(detection_path / 'obj_ids.npy'))
    det_scores = np.load(str(detection_path / 'scores.npy'))
    det_times = np.load(str(detection_path / 'times.npy'))
    assert len(det_scores) == len(pose_scores)
else:
    args.use_pose_score = True
    results_path = Path('data/results')
    det_scene_ids = np.load(str(results_path / f'{args.dataset}-scene_ids.npy'))
    det_view_ids = np.load(str(results_path / f'{args.dataset}-view_ids.npy'))
    det_obj_ids = np.load(str(results_path / f'{args.dataset}-obj_ids.npy'))

scores = pose_scores if args.use_pose_score else det_scores
poses = poses[1 if args.use_refinement else 0]
pose_timings = pose_timings[1 if args.use_refinement else 0]

Rs = poses[:, :3, :3]
ts = poses[:, :3, 3]

img_timings = defaultdict(lambda: 0)

for t, scene_id, view_id in zip(pose_timings, det_scene_ids, det_view_ids):
    img_timings[(scene_id, view_id)] += t

lines = []
for i in range(len(poses)):
    line = ','.join((
        str(det_scene_ids[i]),
        str(det_view_ids[i]),
        str(det_obj_ids[i]),
        str(scores[i]),
        ' '.join((str(v) for v in Rs[i].reshape(-1))),
        ' '.join((str(v) for v in ts[i])),
        f'{det_times[i] + img_timings[(det_scene_ids[i], det_view_ids[i])]}\n',
    ))
    lines.append(line)

if args.use_refinement:
    name += '-refine'
if args.use_pose_score:
    name += '-pose-score'

with open(
        f'data/results/{name}_{args.dataset}-{config[args.dataset].test_folder}.csv'
        , 'w'
) as f:
    f.writelines(lines)
