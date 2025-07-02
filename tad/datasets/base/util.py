import numpy as np


def filter_same_annotation(annotation):
    gt_segments = []
    gt_labels = []
    gt_both = []
    for gt_segment, gt_label in zip(
        annotation["gt_segments"].tolist(), annotation["gt_labels"].tolist()
    ):
        if (gt_segment, gt_label) not in gt_both:
            gt_segments.append(gt_segment)
            gt_labels.append(gt_label)
            gt_both.append((gt_segment, gt_label))
        else:
            continue

    annotation = dict(
        gt_segments=np.array(gt_segments, dtype=np.float32),
        gt_labels=np.array(gt_labels, dtype=np.int32),
    )
    return annotation
