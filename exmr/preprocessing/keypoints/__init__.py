"""2D whole-body keypoints (COCO-63, see mappings.py) from Sapiens2 or RTMW."""

# keypoints with a raw score above this count as visible (RTMW scores are SimCC values, not
# probabilities); used for the hand boxes, and for RTMW also as the body keypoint visibility
VISIBILITY_THRESHOLD = {"sapiens2": 0.5, "rtmw": 5.5}


def detect_keypoints(backend, checkpoint, frame_dir, bbx_xyxy, **kwargs):
    """(L, 63, 3) [x, y, raw score] keypoints of the person in the boxes bbx_xyxy (L, 4)."""
    if backend == "sapiens2":
        from exmr.preprocessing.keypoints.sapiens2 import Sapiens2Keypoints as Model
    elif backend == "rtmw":
        from exmr.preprocessing.keypoints.rtmw import RTMWKeypoints as Model
    else:
        raise ValueError(f"Unknown pose backend: {backend}")
    return Model(checkpoint, **kwargs)(frame_dir, bbx_xyxy)
