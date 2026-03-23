import os
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset


class SpeechCommandsDataset(Dataset):
    def __init__(
        self, root_dir: str, transform: torchvision.transforms = None
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.labels = os.listdir(root_dir)
        for label in self.labels:
            label_dir = os.path.join(root_dir, label)
            for file in os.listdir(label_dir):
                self.file_list.append((os.path.join(label_dir, file), label))

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        file_path, label = self.file_list[idx]
        audio, sr = torchaudio.load(file_path)
        if self.transform:
            audio = self.transform(audio)
        label_idx = self.labels.index(label)
        return audio, label_idx
