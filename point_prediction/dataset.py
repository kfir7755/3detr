import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import trimesh
from sklearn.neighbors import KNeighborsClassifier
from stl import mesh

# Monkey-patch np.int to work like the built-in int (restoring old behavior)
np.bool = bool

# --- Dataset Class ---
class FiveSegmentMeshDataset(Dataset):
    def __init__(self, root: str, normalize: bool = True, cache_knn: bool = True):
        super().__init__()
        self.root = root
        self.samples = sorted([d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d) and
                               os.path.exists(os.path.join(d, 'F.mrk.json'))])
        self.normalize = normalize
        self.cache_knn = cache_knn
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def _get_knn_model(self, sample_dir: str, mean=0, max_l2: float = 1.0):
        cache_path = os.path.join(sample_dir, 'knn_model_cache_old_np_ver.pkl')
        if self.cache_knn and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        X = []
        y = []
        for i in range(1, 6):
            seg_path = os.path.join(sample_dir, 'auto_segmentation_registered-models', f'Segment_{i}.stl')
            if not os.path.exists(seg_path):
                raise FileNotFoundError(f'{seg_path} not found')

            segment_mesh = mesh.Mesh.from_file(seg_path)
            segment_vertices = segment_mesh.vectors.reshape(-1, 3)
            unique_vertices = (np.unique(segment_vertices, axis=0) - mean) / max_l2
            X.extend(unique_vertices)
            y.extend([i] * len(unique_vertices))

        X = np.array(X)
        y = np.array(y)

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X, y)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        return model

    def __getitem__(self, idx: int):
        if idx in self.cache:
            return self.cache[idx]

        sample_dir = self.samples[idx]
        mesh_path = os.path.join(sample_dir, "simplified_mesh_2048.stl")

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"{mesh_path} not found")

        trimesh_mesh = trimesh.load(mesh_path, force='mesh')

        vertices = np.asarray(trimesh_mesh.vertices, dtype=np.float32)
        normals = np.asarray(trimesh_mesh.vertex_normals, dtype=np.float32)

        if self.normalize:
            vertices_mean = np.mean(vertices, axis=0)
            vertices -= vertices_mean
            max_l2 = np.max(np.linalg.norm(vertices, axis=1))
            if max_l2 < 1e-8: max_l2 = 1.0
            vertices /= max_l2
            knn_model = self._get_knn_model(sample_dir, vertices_mean, max_l2)
        else:
            knn_model = self._get_knn_model(sample_dir)
            max_l2 = 1.0
            vertices_mean = np.zeros(3, dtype=np.float32)

        vertex_labels = np.array(knn_model.predict(vertices)) - 1

        b = np.zeros((2048, 5), dtype=np.float32)
        b[np.arange(vertex_labels.size), vertex_labels] = 1

        data = np.zeros((2048, 11), dtype=np.float32)
        data[:, :3] = vertices
        data[:, 3:6] = normals
        data[:, 6:] = b

        lm_path = os.path.join(sample_dir, 'landmarks.npy')
        landmarks = np.load(lm_path).astype(np.float32)

        if self.normalize:
            landmarks -= vertices_mean
            landmarks /= max_l2

        item = (torch.from_numpy(data), torch.from_numpy(landmarks), torch.from_numpy(vertices_mean), float(max_l2))
        self.cache[idx] = item
        return item


def collate_fn(batch):
    data_list, lm_list, mean_list, max_l2_list = zip(*batch)
    return {
        "point_clouds": torch.stack(data_list),
        "targets": torch.stack(lm_list),
        "means": torch.stack(mean_list),
        "max_l2s": torch.tensor(max_l2_list)
    }