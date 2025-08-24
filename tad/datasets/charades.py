import numpy as np
from copy import deepcopy
from .base import PaddingDataset, filter_same_annotation
from .builder import DATASETS
import json


@DATASETS.register_module()
class CharadesPaddingDataset(PaddingDataset):
    def get_video_info(self, ann_file, video_name):
        with open(ann_file, "r") as f:
            anno_database = json.load(f)["database"]
        return anno_database.get(video_name, {})

    def get_gt(self, ann_file, video_name, thresh=0.0):
        gt_segment = []
        gt_label = []
        video_info = self.get_video_info(ann_file, video_name)
        for anno in video_info.get("annotations", []):
            gt_start = int(anno["segment"][0] * self.fps)
            gt_end = int(anno["segment"][1] * self.fps)
            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None

        annotation = dict(
            gt_segments=np.array(gt_segment, dtype=np.float32),
            gt_labels=np.array(gt_label, dtype=np.int32),
        )
        return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name_exo = self.data_list[index]
        video_info_exo = self.get_video_info(self.ann_file_exo, video_name_exo)
        video_anno_exo = self.get_gt(self.ann_file_exo, video_name_exo)

        # get ego view from exo view and pass the the same pipeline
        video_name_ego = video_name_exo + "EGO"
        video_info_ego = self.get_video_info(self.ann_file_ego, video_name_ego)
        video_anno_ego = self.get_gt(self.ann_file_ego, video_name_ego)

        if video_anno_exo is not None:
            video_anno_exo = deepcopy(video_anno_exo)  # avoid modify the original dict
            video_anno_exo["gt_segments"] = (
                video_anno_exo["gt_segments"] - self.offset_frames
            )
            video_anno_exo["gt_segments"] = (
                video_anno_exo["gt_segments"] / self.snippet_stride
            )

        if video_anno_ego is not None:
            video_anno_ego = deepcopy(video_anno_ego)  # avoid modify the original dict
            video_anno_ego["gt_segments"] = (
                video_anno_ego["gt_segments"] - self.offset_frames
            )
            video_anno_ego["gt_segments"] = (
                video_anno_ego["gt_segments"] / self.snippet_stride
            )

        feat_exo = self.pipeline(
            dict(
                video_name=video_name_exo,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=self.fps,
                duration=video_info_exo["duration"],
                offset_frames=self.offset_frames,
                **video_anno_exo,  # unpack -> gt_segments, gt_labels
            )
        )

        feat_ego = self.pipeline(
            dict(
                video_name=video_name_ego,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=self.fps,
                duration=video_info_ego["duration"],
                offset_frames=self.offset_frames,
                **video_anno_ego,  # unpack -> gt_segments, gt_labels
            )
        )

        result = dict(
            inputs_exo=feat_exo["inputs"],
            masks_exo=feat_exo["masks"],
            metas_exo=feat_exo["metas"],
            inputs_ego=feat_ego["inputs"],
            masks_ego=feat_ego["masks"],
            metas_ego=feat_ego["metas"],
        )

        # Add optional keys only if present
        if "gt_segments" in feat_exo:
            result["gt_segments_exo"] = feat_exo["gt_segments"]
        if "gt_labels" in feat_exo:
            result["gt_labels_exo"] = feat_exo["gt_labels"]
        if "gt_segments" in feat_ego:
            result["gt_segments_ego"] = feat_ego["gt_segments"]
        if "gt_labels" in feat_ego:
            result["gt_labels_ego"] = feat_ego["gt_labels"]

        return result
