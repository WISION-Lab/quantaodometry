import copyreg
import functools
import itertools
import tempfile
from pathlib import Path

import compress_pickle as pickle
import cv2
import more_itertools as mitertools
import networkx as nx
import numpy as np
import torch
import torchvision.transforms.functional as TF
from numpy import linalg as LA
from sqlitedict import SqliteDict
from tqdm.auto import tqdm, trange

from quantaodometry.homography import (homography_from_params,
                                       homography_interp, meshgrid_points,
                                       params_from_homography, sample_frames,
                                       warp_points)

from .utils import (cyclic_pairwise, downsample2x, img_to_tensor, to_tensor,
                    torch_grad)

###########################################################
#                       Utilities                         #
###########################################################


# Enable cv2 keypoints to be serialized
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

# ###########################################################
#                 Feature-based Methods                   #
###########################################################


def get_keypoints(image, method="", **kwargs):
    """
    Get key points and feature descriptors using method (i.e: 'sift', 'surf', 'brisk', 'orb').
    """
    descriptor = getattr(cv2, f"{method.upper()}_create") or getattr(cv2.xfeatures2d, f"{method.upper()}_create")

    if descriptor is None:
        raise ValueError(f"Expected method to be one of 'sift', 'surf', 'brisk', 'orb', instead got '{method}'")

    return descriptor(**kwargs).detectAndCompute(image, None)


def match_keypoints(featuresA, featuresB, method, ratio=0.75, startegy="bf", n=None):
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING if method.lower() in ("orb", "brisk") else cv2.NORM_L2,
        crossCheck=startegy.lower() == "bf",
    )

    if startegy.lower() == "bf":
        # Match descriptors, sort the features in order of distance.
        matches = matcher.match(featuresA, featuresB)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches if n is None else matches[:n]
    elif startegy.lower() == "knn":
        # Compute the raw matches and ensure the distance is within a certain
        # ratio of each other (i.e. Lowe's ratio test)
        matches = matcher.knnMatch(featuresA, featuresB, 2)

        if not matches:
            raise RuntimeError("No matches found!")

        matches = [m for m, n in matches if m.distance < n.distance * ratio]
        return matches if n is None else np.random.choice(matches, n)
    raise ValueError(f"Strategy {startegy} is unknown, expected either 'bf' or 'knn'.")


def get_homography(kpsA, kpsB, matches, reproj_threshold=4):
    if not len(matches) > 4:
        raise RuntimeError("At least 4 matches are required!")

    # Get corresponding pairs of points
    kpsA = np.array([kp.pt for kp in kpsA])
    kpsB = np.array([kp.pt for kp in kpsB])
    ptsA = np.array([kpsA[m.queryIdx] for m in matches])
    ptsB = np.array([kpsB[m.trainIdx] for m in matches])

    # Estimate the homography between the sets of points
    return cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_threshold)


def compute_homographies(
    patches, method="sift", ratio=0.75, startegy="bf", n=None, wrt=0, reproj_threshold=4, seed=1234, full=False, **kwargs
):
    if seed is not None:
        cv2.setRNGSeed(seed)

    # This computes pairwise homographies
    keypoints, features = zip(*(get_keypoints(p, method=method, **kwargs) for p in patches))

    valid_idxs, invalid_idxs = [], []
    matches, hs, masks = [], [], []
    for i, (f1, f2, k1, k2) in enumerate(zip(features, features[1:], keypoints, keypoints[1:])):
        try:
            match = match_keypoints(f1, f2, method, ratio=ratio, startegy=startegy, n=n)
            h, mask = get_homography(k1, k2, match, reproj_threshold=reproj_threshold)
            valid_idxs.append(i)
            matches.append(match)
            masks.append(mask)
            hs.append(h)
        except RuntimeError:
            invalid_idxs.append(i)
            matches.append(None)
            masks.append(None)
            hs.append(None)

    if not valid_idxs:
        raise RuntimeError("Could not find any matches!")

    if invalid_idxs:
        # Try to fill in the blanks by interpolating, this doesn't really work well if more than one is missing.
        estimated_hs = homography_interp([h for h in hs if h is not None], valid_idxs, mode="lk")(invalid_idxs)
        for i, h in zip(invalid_idxs, estimated_hs):
            hs[i] = h

    invalid_idxs = [i + 1 for i in invalid_idxs]
    valid_idxs = [
        0,
    ] + [i + 1 for i in valid_idxs]
    hs = [np.eye(3), *hs]

    # To project everything down to the `wrt` frame, we accumulate homographies through composition
    hs = list(itertools.accumulate(hs, lambda x, y: np.matmul(y, x)))
    wrt_inv = LA.inv(hs[wrt])
    hs = [h @ wrt_inv for h in hs]

    if full:
        return hs, keypoints, features, matches, valid_idxs, invalid_idxs, masks
    return hs
