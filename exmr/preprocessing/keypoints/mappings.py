"""Keypoint layouts.

ExMR uses "COCO-63": COCO-WholeBody without the face, i.e. 17 body + 6 foot keypoints ("COCO-23")
followed by 20 left-hand and 20 right-hand keypoints. The hand roots are the body wrists (9, 10).
"""

COCO23 = slice(0, 23)  # body + feet, the body network's 2D input
COCO63_LEFT_HAND = [9, *range(23, 43)]  # wrist + 20 hand keypoints
COCO63_RIGHT_HAND = [10, *range(43, 63)]

# COCO-WholeBody (133): 0-22 body + feet, 23-90 face, 91-111 left hand, 112-132 right hand
COCO133_TO_COCO63 = [*range(23), *range(92, 112), *range(113, 133)]

# Sapiens2 keypoint order (https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md)
SAPIENS2_TO_COCO63 = [
    *range(9), 62, 41, *range(9, 21),
    45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58,
    24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37,
]  # fmt: skip
