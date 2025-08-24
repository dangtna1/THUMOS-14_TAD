import torch
import torch.nn as nn

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from ..bricks import Scale, AffineDropPath


class LearnableFusion(nn.Module):
    """Late fusion with learnable alpha for classification and regression."""

    def __init__(
        self, num_classes: int, per_class: bool = False, exo_bias: float = 1.5
    ):
        super().__init__()
        if per_class:
            self.alpha_cls = nn.Parameter(torch.full((num_classes,), exo_bias))
        else:
            self.alpha_cls = nn.Parameter(torch.tensor(exo_bias))
        self.alpha_reg = nn.Parameter(torch.tensor(exo_bias))
        self.per_class = per_class

    def fuse_lists(self, exo_cls_list, ego_cls_list, exo_reg_list, ego_reg_list):
        fused_cls, fused_reg = [], []
        alpha_cls = (
            torch.sigmoid(self.alpha_cls)[None, :, None]
            if self.per_class
            else torch.sigmoid(self.alpha_cls)
        )
        alpha_reg = torch.sigmoid(self.alpha_reg)

        for c_exo, c_ego in zip(exo_cls_list, ego_cls_list):
            fused_cls.append(alpha_cls * c_exo + (1 - alpha_cls) * c_ego)
        for r_exo, r_ego in zip(exo_reg_list, ego_reg_list):
            fused_reg.append(alpha_reg * r_exo + (1 - alpha_reg) * r_ego)

        return fused_cls, fused_reg


@DETECTORS.register_module()
class ActionFormer(SingleStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
        projection_ego=None,
        neck_ego=None,
        rpn_head_ego=None,
        num_classes=10,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
            neck_ego=neck_ego,
            rpn_head_ego=rpn_head_ego,
            projection_ego=projection_ego,
        )
        self.fusion = LearnableFusion(
            num_classes=num_classes, per_class=False, exo_bias=1.5
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))  # learnable loss weight

        n_mha_win_size = self.projection.n_mha_win_size
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + projection.arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + projection.arch[-1])
            self.mha_win_size = n_mha_win_size
        self.max_seq_len = self.projection.max_seq_len

        max_div_factor = 1
        for s, w in zip(rpn_head.prior_generator.strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert (
                self.max_seq_len % stride == 0
            ), f"max_seq_len {self.max_seq_len} must be divisible by stride {stride}"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    # ----------------- Helpers -----------------

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        max_len = (
            self.max_seq_len
            if feat_len < self.max_seq_len
            else (feat_len + (self.max_div_factor - 1))
            // self.max_div_factor
            * self.max_div_factor
        )
        if max_len == feat_len:
            return inputs, masks
        inputs = torch.nn.functional.pad(inputs, [0, max_len - feat_len], value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    @staticmethod
    def _tiou_1d(a, b):
        inter_start = torch.max(a[0], b[0])
        inter_end = torch.min(a[1], b[1])
        inter = torch.clamp(inter_end - inter_start, min=0)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / torch.clamp(union, min=1e-6)

    def _wbf_1d_per_class(
        self,
        props_list,
        scores_list,
        weights=None,
        iou_thr=0.55,
        score_thr=0.001,
        max_dets=None,
    ):
        if weights is None:
            weights = [1.0] * len(props_list)
        props = torch.cat(props_list, dim=0)
        scores = torch.cat(scores_list, dim=0)
        src_ids = []
        for i, s in enumerate(scores_list):
            src_ids.append(torch.full_like(s, i, dtype=torch.long))
        src_ids = torch.cat(src_ids, dim=0)

        keep = scores >= score_thr
        props, scores, src_ids = props[keep], scores[keep], src_ids[keep]
        if props.numel() == 0:
            return props.new_zeros((0, 2)), scores.new_zeros((0,))

        order = torch.argsort(scores, descending=True)
        props, scores, src_ids = props[order], scores[order], src_ids[order]

        fused_props, fused_scores = [], []
        used = torch.zeros(props.shape[0], dtype=torch.bool, device=props.device)
        for i in range(props.shape[0]):
            if used[i]:
                continue
            cluster_inds = [i]
            used[i] = True
            for j in range(i + 1, props.shape[0]):
                if used[j]:
                    continue
                iou = self._tiou_1d(props[i], props[j])
                if iou >= iou_thr:
                    cluster_inds.append(j)
                    used[j] = True
            ci = torch.tensor(cluster_inds, device=props.device)
            c_props = props[ci]
            c_scores = scores[ci]
            c_src = src_ids[ci]
            w = c_scores * torch.tensor(
                [weights[int(s.item())] for s in c_src],
                device=props.device,
                dtype=c_scores.dtype,
            )
            w = torch.clamp(w, min=1e-6)
            start = (c_props[:, 0] * w).sum() / w.sum()
            end = (c_props[:, 1] * w).sum() / w.sum()
            fused_props.append(torch.stack([start, end]))
            fused_scores.append((c_scores * w / w.sum()).sum())
        fused_props = (
            torch.stack(fused_props) if len(fused_props) else props.new_zeros((0, 2))
        )
        fused_scores = (
            torch.stack(fused_scores) if len(fused_scores) else scores.new_zeros((0,))
        )

        # Ensure fused_props is always 2D
        if fused_props.ndim == 1:
            fused_props = fused_props.unsqueeze(0)

        if max_dets is not None and fused_scores.numel() > max_dets:
            sel = torch.topk(fused_scores, k=max_dets).indices
            fused_props = fused_props[sel]
            fused_scores = fused_scores[sel]
        return fused_props, fused_scores

    def _fuse_batch_proposals_scores(
        self,
        exo_props_list,
        exo_scores_list,
        ego_props_list,
        ego_scores_list,
        exo_weight=1.0,
        ego_weight=0.5,
        iou_thr=0.55,
        score_thr=0.001,
    ):
        """
        Fuses exo and ego proposals and scores using WBF per batch and per class.
        Ensures outputs have shape [num_props, 2] and [num_props, num_classes].
        """
        # Ensure lists of tensors
        if not isinstance(exo_props_list, list):
            exo_props_list = [exo_props_list]
        if not isinstance(exo_scores_list, list):
            exo_scores_list = [exo_scores_list]
        if not isinstance(ego_props_list, list):
            ego_props_list = [ego_props_list]
        if not isinstance(ego_scores_list, list):
            ego_scores_list = [ego_scores_list]

        fused_props_batch, fused_scores_batch = [], []
        B = len(exo_props_list)
        C = exo_scores_list[0].shape[-1]

        for b in range(B):
            fused_props_all = []
            fused_scores_all = []

            for c in range(C):
                props_list = [exo_props_list[b], ego_props_list[b]]
                scores_list = [exo_scores_list[b][:, c], ego_scores_list[b][:, c]]
                fused_p, fused_s = self._wbf_1d_per_class(
                    props_list,
                    scores_list,
                    weights=[exo_weight, ego_weight],
                    iou_thr=iou_thr,
                    score_thr=score_thr,
                )

                # Ensure 2D fused_p and fused_s
                if fused_p.ndim == 1:
                    fused_p = fused_p.unsqueeze(0)
                if fused_s.ndim == 0:
                    fused_s = fused_s.unsqueeze(0)

                if fused_p.numel() > 0:
                    fused_props_all.append(fused_p)
                    # Create per-class score vector for each proposal
                    scores_per_class = torch.zeros(
                        (fused_p.shape[0], C), device=fused_s.device
                    )
                    scores_per_class[:, c] = fused_s
                    fused_scores_all.append(scores_per_class)

            if fused_props_all:
                fused_props_batch.append(torch.cat(fused_props_all, dim=0))
                fused_scores_batch.append(torch.cat(fused_scores_all, dim=0))
            else:
                fused_props_batch.append(exo_props_list[b].new_zeros((0, 2)))
                fused_scores_batch.append(exo_scores_list[b].new_zeros((0, C)))

        return fused_props_batch, fused_scores_batch

    # ----------------- Training -----------------

    def forward_train(
        self,
        inputs_exo,
        masks_exo,
        metas_exo,
        inputs_ego=None,
        masks_ego=None,
        metas_ego=None,
        gt_segments_exo=None,
        gt_labels_exo=None,
        gt_segments_ego=None,
        gt_labels_ego=None,
        **kwargs,
    ):
        losses = {}

        # --- EXO stream
        x_exo = self.backbone(inputs_exo) if self.with_backbone else inputs_exo
        x_exo, masks_exo = self.pad_data(x_exo, masks_exo)
        if self.with_projection:
            x_exo, masks_exo = self.projection(x_exo, masks_exo)
        if self.with_neck:
            x_exo, masks_exo = self.neck(x_exo, masks_exo)
        cls_pred_exo, reg_pred_exo = self.rpn_head.forward_features(x_exo, masks_exo)
        points_exo = self.rpn_head.prior_generator(x_exo)
        loss_exo = self.rpn_head.loss_by_predictions(
            cls_pred_exo,
            reg_pred_exo,
            masks_exo,
            points_exo,
            gt_segments=gt_segments_exo,
            gt_labels=gt_labels_exo,
            **kwargs,
        )
        for k, v in loss_exo.items():
            losses[f"exo.{k}"] = v

        # --- EGO stream (if available)
        if (
            inputs_ego is not None
            and masks_ego is not None
            and gt_segments_ego is not None
        ):
            x_ego = self.backbone(inputs_ego) if self.with_backbone else inputs_ego
            x_ego, masks_ego = self.pad_data(x_ego, masks_ego)
            if hasattr(self, "projection_ego"):
                x_ego, masks_ego = self.projection_ego(x_ego, masks_ego)
            elif self.with_projection:
                x_ego, masks_ego = self.projection(x_ego, masks_ego)
            if hasattr(self, "neck_ego"):
                x_ego, masks_ego = self.neck_ego(x_ego, masks_ego)
            elif self.with_neck:
                x_ego, masks_ego = self.neck(x_ego, masks_ego)
            ego_head = getattr(self, "rpn_head_ego", self.rpn_head)
            cls_pred_ego, reg_pred_ego = ego_head.forward_features(x_ego, masks_ego)
            points_ego = ego_head.prior_generator(x_ego)
            loss_ego = ego_head.loss_by_predictions(
                cls_pred_ego,
                reg_pred_ego,
                masks_ego,
                points_ego,
                gt_segments=gt_segments_ego,
                gt_labels=gt_labels_ego,
                **kwargs,
            )
            for k, v in loss_ego.items():
                losses[f"ego.{k}"] = v

        # total cost (weighted by learnable alpha)
        alpha = torch.sigmoid(self.fusion_alpha)
        losses["cost"] = (
            alpha * sum(loss_exo.values()) + (1 - alpha) * sum(loss_ego.values())
            if inputs_ego is not None
            else sum(loss_exo.values())
        )

        # print("loss_exo:", loss_exo)
        # print("loss_ego:", loss_ego)
        # print("losses:", losses)
        return losses

    # ----------------- Inference -----------------

    def forward_test(
        self,
        inputs_exo,
        masks_exo,
        metas_exo,
        inputs_ego=None,
        masks_ego=None,
        metas_ego=None,
        infer_cfg=None,
        **kwargs,
    ):
        # EXO stream
        x_exo = self.backbone(inputs_exo) if self.with_backbone else inputs_exo
        x_exo, masks_exo = self.pad_data(x_exo, masks_exo)
        if self.with_projection:
            x_exo, masks_exo = self.projection(x_exo, masks_exo)
        if self.with_neck:
            x_exo, masks_exo = self.neck(x_exo, masks_exo)
        cls_pred_exo, reg_pred_exo = self.rpn_head.forward_features(x_exo, masks_exo)
        points_exo = self.rpn_head.prior_generator(x_exo)
        exo_props, exo_scores = self.rpn_head.get_valid_proposals_scores(
            points_exo, reg_pred_exo, cls_pred_exo, masks_exo
        )

        # If no ego -> return exo only
        if inputs_ego is None or masks_ego is None:
            return exo_props, exo_scores

        # EGO stream
        x_ego = self.backbone(inputs_ego) if self.with_backbone else inputs_ego
        x_ego, masks_ego = self.pad_data(x_ego, masks_ego)
        if hasattr(self, "projection_ego"):
            x_ego, masks_ego = self.projection_ego(x_ego, masks_ego)
        elif self.with_projection:
            x_ego, masks_ego = self.projection(x_ego, masks_ego)
        if hasattr(self, "neck_ego"):
            x_ego, masks_ego = self.neck_ego(x_ego, masks_ego)
        elif self.with_neck:
            x_ego, masks_ego = self.neck(x_ego, masks_ego)
        ego_head = getattr(self, "rpn_head_ego", self.rpn_head)
        cls_pred_ego, reg_pred_ego = ego_head.forward_features(x_ego, masks_ego)
        points_ego = ego_head.prior_generator(x_ego)
        ego_props, ego_scores = ego_head.get_valid_proposals_scores(
            points_ego, reg_pred_ego, cls_pred_ego, masks_ego
        )

        # Late fusion via WBF
        fused_props_batch, fused_scores_batch = self._fuse_batch_proposals_scores(
            exo_props,
            exo_scores,
            ego_props,
            ego_scores,
            exo_weight=1.0,
            ego_weight=0.5,
        )

        # Return first batch element (assumes batch size 1)
        fused_props_tensor = fused_props_batch[0]
        fused_scores_tensor = fused_scores_batch[0]

        # Ensure 2D even if empty
        if fused_props_tensor.ndim == 1:
            fused_props_tensor = fused_props_tensor.unsqueeze(0)
        if fused_scores_tensor.ndim == 1:
            fused_scores_tensor = fused_scores_tensor.unsqueeze(0)

        # If completely empty, skip NMS
        if fused_props_tensor.numel() == 0:
            return fused_props_tensor, fused_scores_tensor

        return fused_props_tensor, fused_scores_tensor

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if fpn.startswith("backbone"):
                    continue
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    no_decay.add(fpn)
                elif "fusion" in fpn:  # add this line
                    no_decay.add(fpn)

        param_dict = {
            pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],
                "lr": cfg["lr"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "lr": cfg["lr"],
            },
        ]
        return optim_groups
