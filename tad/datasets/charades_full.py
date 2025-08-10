import os
import os.path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
import tqdm


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, ego_id, start, num):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(
            os.path.join(image_dir, ego_id, ego_id + "-" + str(i).zfill(6) + ".jpg")
        )[:, :, [2, 1, 0]]
        h, w, c = img.shape
        if w < 226 or h < 226:
            d = 226.0 - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.0) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(
            os.path.join(image_dir, vid, vid + "-" + str(i).zfill(6) + "x.jpg"),
            cv2.IMREAD_GRAYSCALE,
        )
        imgy = cv2.imread(
            os.path.join(image_dir, vid, vid + "-" + str(i).zfill(6) + "y.jpg"),
            cv2.IMREAD_GRAYSCALE,
        )

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224.0 - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.0) * 2 - 1
        imgy = (imgy / 255.0) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(ann_file, data_path, mode):  # Get vid + its number of frames
    dataset = []

    with open(ann_file, "r") as f:
        data = json.load(f)
    data = data["database"]

    for vid in data.keys():
        if not os.path.exists(os.path.join(data_path, vid)):
            continue

        num_frames = len(os.listdir(os.path.join(data_path, vid)))

        if mode == "flow":
            num_frames = num_frames // 2

        dataset.append((vid, num_frames))

    return dataset


class Charades(Dataset):

    def __init__(self, ann_file, data_path, mode, transforms=None, save_dir=""):

        self.data = make_dataset(ann_file, data_path, mode)
        self.ann_file = ann_file
        self.transforms = transforms
        self.mode = mode
        self.data_path = data_path
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        ego_id, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, ego_id + ".npy")):
            return 0, ego_id

        if self.mode == "rgb":
            imgs = load_rgb_frames(self.data_path, ego_id, 1, nf)
        else:
            imgs = load_flow_frames(self.data_path, ego_id, 1, nf)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), ego_id

    def __len__(self):
        return len(self.data)
