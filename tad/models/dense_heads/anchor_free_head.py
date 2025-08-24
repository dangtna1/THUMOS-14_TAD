import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale


@HEADS.register_module()
class AnchorFreeHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        filter_similar_gt=True,
    ):
        super(AnchorFreeHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt

        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer(
            "loss_normalizer", torch.tensor(loss_normalizer)
        )  # save in the state_dict

        # point generator
        self.prior_generator = build_prior_generator(prior_generator)

        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    # ---------- layers

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_heads()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_heads(self):
        """Initialize predictor layers of the head."""
        self.cls_head = nn.Conv1d(
            self.feat_channels, self.num_classes, kernel_size=3, padding=1
        )
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scale = nn.ModuleList(
            [Scale() for _ in range(len(self.prior_generator.strides))]
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    # ---------- forward helpers

    def forward_features(self, feat_list, mask_list):
        """Per-level heads → raw logits/reg. feat_list: List[[B,C,T_l]], mask_list: List[[B,T_l]]"""
        cls_pred, reg_pred = [], []
        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)
            cls_pred.append(self.cls_head(cls_feat))  # [B, C_cls, T_l]
            reg_pred.append(
                F.relu(self.scale[l](self.reg_head(reg_feat)))
            )  # [B, 2, T_l]
        return cls_pred, reg_pred

    # ---------- fusion-friendly loss

    def loss_by_predictions(
        self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels, **kwargs
    ):
        """Compute losses directly from (possibly fused) predictions.
        Args:
            cls_pred: List[[B, num_classes, T_l]]
            reg_pred: List[[B, 2, T_l]]
            mask_list: List[[B, T_l]] boolean
            points: List[Tensor] from prior_generator(feat_list_cropped)
            gt_segments, gt_labels: as usual
        """
        # Reuse exact same internal logic as in self.losses, but skip feature forward
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)

        # positive mask
        gt_cls = torch.stack(gt_cls)  # [B, sum(T_l), C]
        valid_mask = torch.cat(mask_list, dim=1)  # [B, sum(T_l)]
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        # EMA normalizer
        if self.training:
            self.loss_normalizer = (
                self.loss_normalizer_momentum * self.loss_normalizer
                + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)
            )
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1) classification
        cls_cat = [x.permute(0, 2, 1) for x in cls_pred]  # -> [B, T_l, C]
        cls_cat = torch.cat(cls_cat, dim=1)[valid_mask]  # -> [N_valid, C]
        gt_target = gt_cls[valid_mask]  # -> [N_valid, C]

        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_cat, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        # 2) regression on positives
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = (
            torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)
        )  # [B,2,T_l] per level
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer

        loss_weight = (
            self.loss_weight
            if self.loss_weight > 0
            else cls_loss.detach() / max(reg_loss.item(), 0.01)
        )

        return {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}

    # ---------- vanilla train/test (single-stream)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred, reg_pred = self.forward_features(feat_list, mask_list)
        points = self.prior_generator(feat_list)
        return self.losses(
            cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels
        )

    def forward_test(self, feat_list, mask_list, **kwargs):
        cls_pred, reg_pred = self.forward_features(feat_list, mask_list)
        points = self.prior_generator(feat_list)
        proposals, scores = self.get_valid_proposals_scores(
            points, reg_pred, cls_pred, mask_list
        )
        return proposals, scores

    # ---------- utilities

    def get_refined_proposals(self, points, reg_pred):
        points = torch.cat(points, dim=0)  # [T,4]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)  # [B,T,2]
        start = points[:, 0][None] - reg_pred[:, :, 0] * points[:, 3][None]
        end = points[:, 0][None] + reg_pred[:, :, 1] * points[:, 3][None]
        proposals = torch.stack((start, end), dim=-1)  # [B,T,2]
        return proposals

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list):
        proposals = self.get_refined_proposals(points, reg_pred)  # [B,T,2]
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,C]
        masks = torch.cat(mask_list, dim=1)  # [B,T]
        new_proposals, new_scores = [], []
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])  # [T,2]
            new_scores.append(score[mask])  # [T,C]
        return new_proposals, new_scores

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        # (unchanged, shared by forward_train)
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)

        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        if self.training:
            self.loss_normalizer = (
                self.loss_normalizer_momentum * self.loss_normalizer
                + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)
            )
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        cls_cat = [x.permute(0, 2, 1) for x in cls_pred]
        cls_cat = torch.cat(cls_cat, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_cat, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer

        loss_weight = (
            self.loss_weight
            if self.loss_weight > 0
            else cls_loss.detach() / max(reg_loss.item(), 0.01)
        )
        return {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        gt_cls, gt_reg = [], []

        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            num_gts = gt_segment.shape[0]
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                t_mins = (
                    center_pts - concat_points[:, 3, None] * self.center_sample_radius
                )
                t_maxs = (
                    center_pts + concat_points[:, 3, None] * self.center_sample_radius
                )
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(
                    t_mins, gt_segs[:, :, 0]
                )
                cb_dist_right = (
                    torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                )
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            max_regress_distance = reg_targets.max(-1)[0]
            inside_regress_range = torch.logical_and(
                (max_regress_distance >= concat_points[:, 1, None]),
                (max_regress_distance <= concat_points[:, 2, None]),
            )

            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            min_len, min_len_inds = lens.min(dim=1)

            if self.filter_similar_gt:
                min_len_mask = torch.logical_and(
                    (lens <= (min_len[:, None] + 1e-3)), (lens < float("inf"))
                )
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(
                reg_targets.dtype
            )
            cls_targets = min_len_mask @ gt_label_one_hot
            cls_targets.clamp_(min=0.0, max=1.0)

            reg_targets = reg_targets[range(num_pts), min_len_inds]
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg
