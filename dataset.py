import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        device: str,
        transform: torchaudio.transforms | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform.to(device) if transform else None
        self.file_list = []
        self.labels = os.listdir(root_dir)
        self.device = device
        for label in self.labels:
            label_dir = os.path.join(root_dir, label)
            for file in os.listdir(label_dir):
                self.file_list.append((os.path.join(label_dir, file), label))

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        file_path, label = self.file_list[idx]
        data, _ = sf.read(file_path)
        audio = torch.from_numpy(data).float().to(self.device)
        if self.transform:
            audio = self.transform(audio)
        label_idx = self.labels.index(label)
        return audio, label_idx


if __name__ == "__main__":
    import argparse

    argparse = argparse.ArgumentParser(description="Test SpeechCommandsDataset")
    argparse.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory",
    )
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.parse_args()
    dataset = SpeechCommandsDataset(
        root_dir=args.root_dir, device=device, transform=mel_spec
    )
    print(f"Dataset size: {len(dataset)}")
    audio, label_idx = dataset[0]
    print(f"Audio shape: {audio.shape}, Label index: {label_idx}")
