from torch.utils.data import Dataset
import numpy as np


class MeshDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print(f"[MeshDataset] Created from {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # Create a copy to avoid modifying original data and prevent memory leaks
        data_copy = {}
        for key, value in data.items():
            if hasattr(value, 'clone'):
                data_copy[key] = value.clone()
            else:
                data_copy[key] = value
        return data_copy

    def save(self, path):
        np.savez_compressed(path, self.data, allow_pickle=True)
        print(f"[MeshDataset] Saved {len(self.data)} entries at {path}")

    @classmethod
    def load(cls, path):
        loaded_data = np.load(path, allow_pickle=True)
        data = []
        for item in loaded_data["arr_0"]:
            data.append(item)
        print(f"[MeshDataset] Loaded {len(data)} entries")
        return cls(data)