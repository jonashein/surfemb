import argparse
from pathlib import Path

import trimesh
import trimesh.proximity
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

mesh_folder = Path(args.dataset) / 'models_eval'
even_samples_folder = Path(args.dataset) / 'models_even_surface_samples'
assert even_samples_folder.exists()
normals_folder = Path(args.dataset) / 'models_surface_samples_normals'
normals_folder.mkdir(exist_ok=True, parents=True)

for mesh_fp in tqdm(list(mesh_folder.glob('*.ply'))):
    samples_fp = even_samples_folder / mesh_fp.name
    normals_fp = normals_folder / mesh_fp.name

    mesh = trimesh.load_mesh(mesh_fp)  # type: trimesh.Trimesh
    sample_pts = trimesh.load_mesh(samples_fp).vertices
    face_idx = trimesh.proximity.closest_point(mesh, sample_pts)[-1]
    normals = mesh.face_normals[face_idx]

    pc = trimesh.PointCloud(normals)
    pc.export(normals_fp)
