import torch

# import torch.nn.functional as F
# import torchvision
# import scipy
import numpy as np
from collections.abc import Sequence
from einops import rearrange, reduce

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@PIPELINES.register_module()
class ConvertToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"


@PIPELINES.register_module()
class Rearrange:
    def __init__(self, keys, ops, **kwargs):
        self.keys = keys
        self.ops = ops
        self.kwargs = kwargs

    def __call__(self, results):
        for key in self.keys:
            results[key] = rearrange(results[key], self.ops, **self.kwargs)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys}ops={self.ops})"


@PIPELINES.register_module()
class Collect:
    def __init__(
        self,
        inputs,
        keys=[],
        meta_keys=[
            "video_name",
            "data_path",
            "fps",
            "duration",
            "snippet_stride",
            "window_start_frame",
            "resize_length",
            "window_size",
            "offset_frames",
        ],
    ):
        self.inputs = inputs
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}

        # input key
        data["inputs"] = results[self.inputs]  # [C,T]

        # AutoAugment key: gt_segments, gt_labels, masks
        for key in self.keys:
            if key == "masks" and key not in results.keys():
                results["masks"] = torch.ones(data["inputs"].shape[-1]).bool()
            data[key] = results[key]

        # meta keys
        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                if key in results.keys():
                    meta[key] = results[key]
            data["metas"] = meta

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"keys={self.keys}, meta_keys={self.meta_keys}, "
        )
