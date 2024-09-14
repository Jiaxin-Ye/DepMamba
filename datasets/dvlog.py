from pathlib import Path
from typing import Union, Optional

import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

class DVlog(data.Dataset):
    def __init__(
        self, root: Union[str, Path], fold: str="train", 
        gender: str="both", transform=None, target_transform=None, aug=False
    ):
        self.root = root if isinstance(root, Path) else Path(root)
        self.fold = fold
        self.gender = gender
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

        self.features = []
        self.labels = []
        with open(self.root / "labels.csv", "r") as f:
            for line in f:
                # print(line)
                sample = line.strip().split(",")
                if self.is_sample(sample):
                    s_id = sample[0]
                    s_label = int(sample[1]=="depression")
                    self.labels.append(s_label)

                    v_feature_path = self.root / s_id / f"{s_id}_visual.npy"
                    a_feature_path = self.root / s_id / f"{s_id}_acoustic.npy"
                    v_feature = np.load(v_feature_path)
                    a_feature = np.load(a_feature_path)
                    # concat visual and acoustic features along the 2nd axis
                    T_v, T_a = v_feature.shape[0], a_feature.shape[0]
                    if T_v == T_a:
                        feature = np.concatenate(
                            (v_feature, a_feature), axis=1
                        ).astype(np.float32)
                    else:
                        T = min(T_v, T_a)
                        feature = np.concatenate(
                            (v_feature[:T], a_feature[:T]), axis=1
                        ).astype(np.float32)
                    self.features.append(feature)

                    if self.aug and self.fold=='train':
                        t_length = feature.shape[0]
                        for i in range(5):# if s_label==0 else 4
                            f_length = int(random.random()*t_length)
                            if f_length<500:
                                continue
                            t_start = random.randint(1, t_length-f_length)
                            self.labels.append(s_label)
                            self.features.append(feature[t_start:t_start+f_length,:])
                            # print(feature[t_start:t_start+f_length,:].shape)

        print(f"ALL:{len(self.labels)}, Positive:{np.sum(self.labels)}, Negative:{len(self.labels)-np.sum(self.labels)}")

    def is_sample(self, sample) -> bool:
        gender, fold = sample[3], sample[4]
        if self.gender == "both":
            return fold == self.fold
        return (fold == self.fold) and (gender == self.gender)

    def __getitem__(self, i: int):
        # i = random.randint(0, len(self.labels)-1)
        feature = self.features[i]
        label = self.labels[i]
        if self.transform is not None:
            print("Transform 1")
            feature = self.transform(feature)
        if self.target_transform is not None:
            print("Transform 2")
            label = self.target_transform(label)
        return feature, label

    def __len__(self):
        return len(self.labels)


def _collate_fn(batch):
    # batch: [(feature, label), (feature, label), ...]
    # feature.shape = [T, F], F is fixed, but T varies from sample to sample
    features, labels = zip(*batch)
    padded_features = pad_sequence(
        [torch.from_numpy(f) for f in features], batch_first=True
    )
    padding_mask = (padded_features.sum(dim=-1) != 0).long()
    labels = torch.tensor(labels)
    return padded_features, labels, padding_mask


def get_dvlog_dataloader(
    root: Union[str, Path], fold: str="train", batch_size: int=8, 
    gender: str="both",
    transform=None, target_transform=None, aug=True
):
    """Get dataloader for DVlog dataset.

    Args:
        root (Union[str, Path]): path to the dvlog dataset. Should be something
            like `*/dvlog-dataset`.
        fold (str, optional): train / valid / test. Defaults to "train".
        batch_size (int, optional): Defaults to 8.
        gender (str, optional): m / f / both. Defaults to both.
        transform (optional): Defaults to None.
        target_transform (optional): Defaults to None.

    Returns:
        the dataloader.
    """
    dataset = DVlog(root, fold, gender, transform, target_transform, aug)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, 
        collate_fn=_collate_fn,
        shuffle=(fold=="train"),
    )
    return dataloader


if __name__ == '__main__':
    train_loader = get_dvlog_dataloader(
        "./dataset/dvlog", "train"
    )
    print(f"train_loader: {len(train_loader.dataset)} samples")
    valid_loader = get_dvlog_dataloader(
        "./dataset/dvlog", "valid"
    )
    print(f"valid_loader: {len(valid_loader.dataset)} samples")
    test_loader = get_dvlog_dataloader(
        "./dataset/dvlog", "test"
    )
    print(f"test_loader: {len(test_loader.dataset)} samples")

    b1 = next(iter(train_loader))[0]
    print(f"A train_loader batch: shape={b1.shape}, dtype={b1.dtype}")
    b2 = next(iter(valid_loader))[0]
    print(f"A valid_loader batch: shape={b2.shape}, dtype={b2.dtype}")
    b3 = next(iter(test_loader))[0]
    print(f"A test_loader batch: shape={b3.shape}, dtype={b3.dtype}")
