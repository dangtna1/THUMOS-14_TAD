import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="rgb", help="rgb or flow")
parser.add_argument("--ann_file", required=True, type=str)
parser.add_argument("--save_dir", required=True, type=str)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import transforms
from tad.datasets import CenterCrop


import numpy as np

from pytorch_i3d import InceptionI3d

from tad.datasets import Charades
import tqdm


def run(
    max_steps=64e3,
    mode="rgb",
    data_path="data/charades/CharadesEgo_v1_rgb",
    ann_file="data/charades/annotations/exo_ego_charades.json",
    batch_size=1,
    load_model="models/rgb_charades.pt",
    save_dir="data/charades/features_ego/",
):
    # setup dataset
    test_transforms = transforms.Compose([CenterCrop(224)])

    dataset = Charades(ann_file, data_path, mode, test_transforms, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    # setup the model
    if mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    # Load pre-trained model
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model, weights_only=True))

    # Start extracting
    i3d.train(False)  # Set model to evaluate mode

    # Iterate over data.
    for data in tqdm.tqdm(dataloader):
        # get the inputs
        inputs, name = data
        if os.path.exists(os.path.join(save_dir, name[0] + ".npy")):
            continue

        b, c, t, h, w = inputs.shape  # b = batch = 1
        if t > 1600:
            features = []
            for start in range(1, t - 56, 1600):
                end = min(t - 1, start + 1600 + 56)
                start = max(1, start - 48)
                ip = Variable(
                    torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda(),
                    volatile=True,
                )
                features.append(
                    i3d.extract_features(ip)
                    .squeeze(0)
                    .permute(1, 2, 3, 0)
                    .data.cpu()
                    .numpy()
                )
            np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            print("Oh no, what happened?")
        else:
            with torch.no_grad():
                inputs = inputs.to("cpu")  # make sure inputs are on CPU
                features = i3d.extract_features(inputs)
                f = (
                    features.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                )  # shape (T, 1, 1, 1024)
                f = f.reshape(f.shape[0], -1)  # shape (T, 1024)
                np.save(os.path.join(save_dir, name[0]), f)


if __name__ == "__main__":
    # need to add argparse
    run(mode=args.mode, ann_file=args.ann_file, save_dir=args.save_dir)
