import torch
from ..utils.post_processing import load_predictions, save_predictions


class BaseDetector(torch.nn.Module):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

    def forward(
        self,
        inputs_exo,
        masks_exo,
        metas_exo,
        inputs_ego,
        masks_ego,
        metas_ego,
        gt_segments_exo=None,
        gt_labels_exo=None,
        gt_segments_ego=None,
        gt_labels_ego=None,
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            return self.forward_train(
                inputs_exo=inputs_exo,
                masks_exo=masks_exo,
                metas_exo=metas_exo,
                inputs_ego=inputs_ego,
                masks_ego=masks_ego,
                metas_ego=metas_ego,
                gt_segments_exo=gt_segments_exo,
                gt_labels_exo=gt_labels_exo,
                gt_segments_ego=gt_segments_ego,
                gt_labels_ego=gt_labels_ego,
                **kwargs,
            )
        else:
            return self.forward_detection(
                input_exo=inputs_exo,
                masks_exo=masks_exo,
                metas_exo=metas_exo,
                input_ego=inputs_ego,
                masks_ego=masks_ego,
                metas_ego=metas_ego,
                infer_cfg=infer_cfg,
                post_cfg=post_cfg,
                **kwargs,
            )

    def forward_detection(
        self,
        input_exo,
        masks_exo,
        metas_exo,
        input_ego=None,
        masks_ego=None,
        metas_ego=None,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        # step1: inference the model (load or run)
        if infer_cfg is not None and getattr(
            infer_cfg, "load_from_raw_predictions", False
        ):
            predictions = load_predictions(metas_exo, infer_cfg)
        else:
            predictions = self.forward_test(
                inputs_exo=input_exo,
                masks_exo=masks_exo,
                metas_exo=metas_exo,
                inputs_ego=input_ego,
                masks_ego=masks_ego,
                metas_ego=metas_ego,
                infer_cfg=infer_cfg,
                **kwargs,
            )
            if infer_cfg is not None and getattr(
                infer_cfg, "save_raw_prediction", False
            ):
                save_predictions(predictions, metas_exo, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas_exo, post_cfg, **kwargs)
        return results
