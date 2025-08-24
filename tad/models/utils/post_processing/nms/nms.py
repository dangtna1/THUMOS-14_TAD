# Functions for 1D NMS, modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import torch

import nms_1d_cpu


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, segs, scores, cls_idxs, iou_threshold, min_score, max_num):
        # Normalize shapes
        if segs.numel() == 0 or scores.numel() == 0:
            return (
                segs.new_zeros((0, 2)),
                scores.new_zeros((0,)),
                cls_idxs.new_zeros((0,), dtype=torch.long),
            )

        # Ensure segs is 2D (N, 2), scores is (N,), cls_idxs is (N,)
        if segs.dim() == 1:
            segs = segs.unsqueeze(0)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        if cls_idxs.dim() == 0:
            cls_idxs = cls_idxs.unsqueeze(0)

        # Move to CPU for the C++ op (but remember original device)
        orig_device = segs.device
        segs_cpu = segs.contiguous().cpu()
        scores_cpu = scores.contiguous().cpu()
        cls_cpu = cls_idxs.contiguous().cpu()

        # apply vanilla NMS (expects CPU tensors)
        inds = nms_1d_cpu.nms(
            segs_cpu.contiguous(),
            scores_cpu.contiguous(),
            iou_threshold=float(iou_threshold),
        )

        # cap by max_num
        if max_num > 0:
            inds = inds[: min(max_num, len(inds))]

        # gather results and move back to original device
        sorted_segs = segs_cpu[inds].to(orig_device)
        sorted_scores = scores_cpu[inds].to(orig_device)
        sorted_cls_idxs = cls_cpu[inds].to(orig_device)

        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        segs,
        scores,
        cls_idxs,
        iou_threshold,
        sigma,
        min_score,
        method,
        max_num,
        t1,
        t2,
    ):
        # --- Quick empty handling
        if segs is None or scores is None or segs.numel() == 0 or scores.numel() == 0:
            return (
                segs.new_zeros((0, 2)),
                scores.new_zeros((0,)),
                cls_idxs.new_zeros((0,), dtype=torch.long),
            )

        # --- Normalize common collapsed shapes into (N,2) / (N,) forms
        # If segs is 1-D with 2 elements -> treat as single proposal [start, end]
        if segs.dim() == 1 and segs.numel() == 2:
            segs = segs.view(1, 2)

        # If segs is (2,1) (i.e., column) convert to (1,2)
        if segs.dim() == 2 and segs.size(0) == 2 and segs.size(1) == 1:
            segs = segs.t().contiguous().view(1, 2)

        # If segs is (1,1) but has 2 elements flattened somewhere, try a safe reshape:
        if segs.dim() == 1 and segs.numel() == 1:
            # cannot reconstruct start/end from a single value — fallback
            return (
                segs.new_zeros((0, 2)),
                scores.new_zeros((0,)),
                cls_idxs.new_zeros((0,), dtype=torch.long),
            )

        # Ensure scores shape is (N,)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        if scores.dim() > 1 and scores.size(1) == 1:
            scores = scores.view(-1)

        # Ensure cls_idxs is 1D
        if cls_idxs is not None and cls_idxs.dim() == 0:
            cls_idxs = cls_idxs.unsqueeze(0)

        # Final guard: ensure segs is (N,2)
        if not (segs.dim() == 2 and segs.size(1) == 2):
            # Attempt final reshape if numel matches 2 * N
            if segs.numel() % 2 == 0:
                segs = segs.view(-1, 2)
            else:
                # give up safely
                return (
                    segs.new_zeros((0, 2)),
                    scores.new_zeros((0,)),
                    cls_idxs.new_zeros((0,), dtype=torch.long),
                )

        # Keep device to move results back
        orig_device = segs.device

        # Move to CPU for the C++ op (and ensure contiguous)
        segs_cpu = segs.contiguous().cpu()
        scores_cpu = scores.contiguous().cpu()

        # allocate dets on CPU (N x 3)
        dets = torch.empty((segs_cpu.size(0), 3), device="cpu", dtype=segs_cpu.dtype)

        # Try native softnms — guard with try/except to avoid uncaught C++ errors
        try:
            inds = nms_1d_cpu.softnms(
                segs_cpu,
                scores_cpu,
                dets,
                iou_threshold=float(iou_threshold),
                sigma=float(sigma),
                min_score=float(min_score),
                method=int(method),
                t1=float(t1),
                t2=float(t2),
            )

            # cap by max number
            if max_num > 0:
                n_segs = min(len(inds), max_num)
            else:
                n_segs = len(inds)

            sorted_segs = dets[:n_segs, :2].to(orig_device)
            sorted_scores = dets[:n_segs, 2].to(orig_device)

            if cls_idxs.numel() > 0:
                cls_cpu = cls_idxs.contiguous().cpu()
                sorted_cls_idxs = cls_cpu[inds][:n_segs].to(orig_device)
            else:
                sorted_cls_idxs = cls_idxs.new_zeros((0,), dtype=torch.long)

            return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()

        except Exception as e:
            # If native softnms fails for any reason, fallback to a safe deterministic behavior:
            # - filter by min_score
            # - sort by score desc
            # - return top-k segments and their class indices
            # Log the problem (you can comment out the print in production)
            print(
                f"[SoftNMSop] softnms failed with exception: {e}. Falling back to sort+topk."
            )

            keep_mask = scores_cpu >= float(min_score)
            if keep_mask.numel() == 0 or not keep_mask.any():
                return (
                    segs.new_zeros((0, 2)),
                    scores.new_zeros((0,)),
                    cls_idxs.new_zeros((0,), dtype=torch.long),
                )

            kept_segs = segs_cpu[keep_mask]
            kept_scores = scores_cpu[keep_mask]
            if cls_idxs.numel() > 0:
                kept_cls = cls_idxs.contiguous().cpu()[keep_mask]
            else:
                kept_cls = cls_idxs.new_zeros((kept_scores.size(0),), dtype=torch.long)

            # sort
            sorted_vals, idxs = torch.sort(kept_scores, descending=True)
            if max_num > 0:
                idxs = idxs[:max_num]
                sorted_vals = sorted_vals[:max_num]

            sorted_segs = kept_segs[idxs].to(orig_device)
            sorted_scores = sorted_vals.to(orig_device)
            sorted_cls_idxs = kept_cls[idxs].to(orig_device)

            return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """
    blur localization results by incorporating side segs.
    this is known as bounding box voting in object detection literature.
    slightly boost the performance around iou_threshold
    """

    # *_segs : N_i x 2, all_scores: N,
    # apply offset
    offset_scores = all_scores + score_offset

    # computer overlap between nms and all segs
    # construct the distance matrix of # N_nms x # N_all
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    # compute intersection
    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right - left).clamp(min=0)

    # lens of all segments
    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]

    # iou
    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    # get neighbors (# N_nms x # N_all) / weights
    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :]
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    refined_segs = seg_weights @ all_segs

    return refined_segs


def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold=0.0,  # does not matter when use soft nms
    min_score=0.0,
    max_seg_num=100,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.0,  # set 0 to disable
    method=2,  # 0: vanilla nms, 1: linear, 2: gaussian, 3: improved gaussian
    t1=0,  # only used in improved gaussian for better recall
    t2=0,  # only used in improved gaussian for better recall
):
    # --- Robust input normalization (INSERTED)
    # make sure the inputs are float
    segs = segs.float()
    scores = scores.float()

    # Normalize shapes: segs -> (N,2), scores -> (N,), cls_idxs -> (N,)
    if segs.dim() == 1:
        segs = segs.unsqueeze(0)
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
    if cls_idxs is not None and cls_idxs.dim() == 0:
        cls_idxs = cls_idxs.unsqueeze(0)

    # handle empty quickly
    num_segs = segs.shape[0]
    if num_segs == 0:
        return (
            torch.zeros([0, 2]),
            torch.zeros([0]),
            torch.zeros(
                [0], dtype=cls_idxs.dtype if cls_idxs is not None else torch.long
            ),
        )
    # --- End inserted guard

    if multiclass:
        # multiclass nms: apply nms on each class independently
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            # soft_nms vs nms
            if use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    sigma,
                    min_score,
                    method,
                    max_seg_num,
                    t1,
                    t2,
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs = NMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    min_score,
                    max_seg_num,
                )
            # disable seg voting for multiclass nms, no sufficient segs

            # fill in the class index
            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        # cat the results
        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)

    else:
        # class agnostic
        if use_soft_nms:
            new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                segs,
                scores,
                cls_idxs,
                iou_threshold,
                sigma,
                min_score,
                method,
                max_seg_num,
                t1,
                t2,
            )
        else:
            new_segs, new_scores, new_cls_idxs = NMSop.apply(
                segs,
                scores,
                cls_idxs,
                iou_threshold,
                min_score,
                max_seg_num,
            )
        # seg voting
        if voting_thresh > 0:
            new_segs = seg_voting(new_segs, segs, scores, voting_thresh)

    # sort based on scores and return
    # truncate the results based on max_seg_num
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    # needed for multiclass NMS
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
