"""
dataset.py

PyTorch Dataset for loading Robosuite Lift demonstrations.

Each sample returns:
    {
        "image": Tensor (3, H, W),
        "text":  string,
        "action": Tensor (7,)
    }

Ground-truth state is NOT exposed.
Only RGB + text + action are used.
"""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class LiftDataset(Dataset):
    def __init__(self, hdf5_path, image_size=224):
        """
        Args:
            hdf5_path (str): Path to demo_data.h5
            image_size (int): Resize images to this size
        """

        self.hdf5_path = hdf5_path
        self.image_size = image_size

        # Open HDF5 file once (kept open during dataset lifetime)
        self.file = h5py.File(self.hdf5_path, "r")
        self.data_group = self.file["data"]

        # Build (demo_key, timestep) index list
        self.indices = []

        for demo_key in self.data_group.keys():
            demo = self.data_group[demo_key]
            num_steps = demo["images"].shape[0]

            for t in range(num_steps):
                self.indices.append((demo_key, t))

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),                    # (H,W,C) → (C,H,W), scaled to [0,1]
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns one timestep sample.
        """

        demo_key, timestep = self.indices[idx]
        demo = self.data_group[demo_key]

        # Load image (H, W, 3)
        image = demo["images"][timestep]

        # Load action (7,)
        action = demo["actions"][timestep]

        # Load instruction (stored once per episode)
        text = demo["text"][()]

        # Decode bytes → string (if needed)
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        # Apply transforms
        image = self.transform(image)
        action = torch.tensor(action, dtype=torch.float32)

        return {
            "image": image,
            "text": text,
            "action": action
        }

    def close(self):
        """Call this when done using dataset."""
        if self.file:
            self.file.close()
