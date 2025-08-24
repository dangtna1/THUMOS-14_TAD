import torch
from ..builder import (
    DETECTORS,
    build_backbone,
    build_projection,
    build_head,
    build_neck,
)
from .base import BaseDetector

from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """
    Base class for single-stage detectors which should not have roi_extractors.
    """

    def __init__(
        self,
        backbone=None,
        projection=None,
        neck=None,
        rpn_head=None,
        projection_ego=None,
        neck_ego=None,
        rpn_head_ego=None,
    ):
        super(SingleStageDetector, self).__init__()

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

        if projection_ego is not None:
            self.projection_ego = build_projection(projection_ego)

        if neck_ego is not None:
            self.neck_ego = build_neck(neck_ego)

        if rpn_head_ego is not None:
            self.rpn_head_ego = build_head(rpn_head_ego)

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    @property
    def with_projection_ego(self):
        """bool: whether the detector has ego projection"""
        return hasattr(self, "projection_ego") and self.projection_ego is not None

    @property
    def with_neck_ego(self):
        """bool: whether the detector has ego neck"""
        return hasattr(self, "neck_ego") and self.neck_ego is not None

    @property
    def with_rpn_head_ego(self):
        """bool: whether the detector has ego localization head"""
        return hasattr(self, "rpn_head_ego") and self.rpn_head_ego is not None

    def forward_train(
        self,
        exo_inputs,
        exo_masks,
        ego_inputs,
        ego_masks,
        metas,
        gt_segments,
        gt_labels,
        **kwargs
    ):
        losses = dict()
        if self.with_backbone:
            x_exo = self.backbone(exo_inputs, exo_masks)
        else:
            x_exo = exo_inputs

        if self.with_projection:
            x_exo, exo_masks = self.projection(x_exo, exo_masks)

        if self.with_neck:
            x_exo, exo_masks = self.neck(x_exo, exo_masks)

        # if self.with_rpn_head:
        #     rpn_losses = self.rpn_head.forward_train(
        #         x,
        #         exo_masks,
        #         gt_segments=gt_segments,
        #         gt_labels=gt_labels,
        #         **kwargs,
        #     )
        #     losses.update(rpn_losses)

        # get exo predictions (raw logits/reg) via head helper
        if self.with_rpn_head:
            exo_cls_logits, exo_reg_preds = self.rpn_head.forward_features(
                x_exo, exo_masks
            )
        else:
            exo_cls_logits = exo_reg_preds = None

        # --- EGO stream (if modules provided)
        ego_available = hasattr(self, "projection_ego") or hasattr(self, "rpn_head_ego")
        if ego_available:
            # expecting caller to pass ego inputs and ego masks in kwargs
            if ego_inputs is None:
                # fallback: zeros same shape as exo stream
                ego_inputs = torch.zeros_like(inputs)
                ego_masks = masks
            if self.with_backbone:
                x_ego = self.backbone(
                    ego_inputs
                )  # optionally: use a different backbone if you want
            else:
                x_ego = ego_inputs

            # # pad ego stream separately
            # x_ego, masks_ego = self.pad_data(x_ego, ego_masks)

            if hasattr(self, "projection_ego"):
                x_ego, masks_ego = self.projection_ego(x_ego, masks_ego)
            elif self.with_projection:  # reuse exo projection if desired
                x_ego, masks_ego = self.projection(x_ego, masks_ego)

            if hasattr(self, "neck_ego"):
                x_ego, masks_ego = self.neck_ego(x_ego, masks_ego)
            elif self.with_neck:
                x_ego, masks_ego = self.neck(x_ego, masks_ego)

            if self.with_rpn_head_ego:
                ego_cls_logits, ego_reg_preds = self.rpn_head_ego.forward_features(
                    x_ego, masks_ego
                )
            else:
                # reuse same head weights (not recommended but supported)
                ego_cls_logits, ego_reg_preds = self.rpn_head.forward_features(
                    x_ego, masks_ego
                )

        # --- Fusion

        if ego_cls_logits is None:
            # single-stream behavior
            losses_rpn = self.rpn_head.forward_train(
                x_exo, masks_exo, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs
            )
            losses.update(losses_rpn)
        else:
            # Example fusion: trainable alpha per-class (scalar) or simple average
            # Simple average:
            fused_cls = (exo_cls_logits + ego_cls_logits) / 2.0
            fused_reg = (exo_reg_preds + ego_reg_preds) / 2.0

            # OR trainable scalar alpha:
            # if not hasattr(self, "fusion_alpha"):
            #     self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
            # alpha = torch.sigmoid(self.fusion_alpha)
            # fused_cls = alpha * exo_cls_logits + (1-alpha) * ego_cls_logits
            # fused_reg = alpha * exo_reg_preds + (1-alpha) * ego_reg_preds

            # compute loss from fused predictions using head helper
            # Use exo head's loss_by_predictions (or create a shared loss module)
            fused_losses = self.rpn_head.loss_by_predictions(
                fused_cls,
                fused_reg,
                masks_exo,  # why only need exo masks?
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs
            )
            losses.update(fused_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        print("losses >>> ", losses)
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks)
        else:
            rpn_proposals = rpn_scores = None

        predictions = rpn_proposals, rpn_scores
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        rpn_proposals, rpn_scores = predictions
        # rpn_proposals: [B, K, 2]
        # rpn_scores: [B, K, num_classes] after sigmoid

        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = rpn_proposals[i].detach().cpu()  # [N,2]
            scores = rpn_scores[i].detach().cpu()  # [N,num_classes]

            # Handle empty proposals
            if segments.numel() == 0 or scores.numel() == 0:
                results[metas[i]["video_name"]] = []
                continue

            num_classes = scores.shape[-1]

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0], dtype=torch.long)
            else:
                pred_prob = scores.flatten()  # [N*num_classes]

                # 1. Keep seg with confidence score > threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top-k highest scoring
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                if num_topk == 0:
                    results[metas[i]["video_name"]] = []
                    continue

                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # 3. Gather predicted proposals and class labels
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)

                if pt_idxs.numel() == 0:
                    segments = segments.new_zeros((0, 2))
                    scores = scores.new_zeros((0,))
                    labels = cls_idxs.new_zeros((0,), dtype=torch.long)
                else:
                    segments = segments[pt_idxs]
                    scores = pred_prob
                    labels = cls_idxs

            # Apply NMS only if there are proposals
            if segments.numel() > 0 and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(
                    segments, scores, labels, **post_cfg.nms
                )

            video_id = metas[i]["video_name"]

            # Convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # Merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            results[video_id] = results_per_video

        return results
