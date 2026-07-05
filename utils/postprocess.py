import cv2
import numpy as np
from scipy import ndimage


def _odd_kernel_size(short_side, ratio):
    size = max(3, int(round(short_side * ratio)))
    if size % 2 == 0:
        size += 1
    return size


def _expand_seeded_components(candidate, seed, min_expand_area):
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(candidate.astype("uint8"), connectivity=8)
    if num_labels <= 1:
        return candidate.astype(bool)
    kept = np.zeros_like(candidate, dtype=bool)
    seed = seed.astype(bool)
    for label in range(1, num_labels):
        component = labels == label
        if np.any(component & seed):
            if int(component.sum()) >= min_expand_area:
                kept |= component
            else:
                kept |= component & seed
    return kept


def _remove_small_components(mask, min_area, preserve=None):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    kept = np.zeros_like(mask, dtype=bool)
    preserve = np.zeros_like(mask, dtype=bool) if preserve is None else preserve.astype(bool)
    for label in range(1, num_labels):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] >= min_area or np.any(component & preserve):
            kept |= component
    return kept


def refine_corrosion_mask(
    prob_np,
    threshold=0.65,
    low_threshold=0.25,
    high_threshold=None,
    close_kernel_ratio=0.012,
    fill_holes=True,
    min_expand_area_ratio=0.01,
    min_component_area_ratio=0.0005,
):
    raw_mask = prob_np >= threshold
    high_threshold = threshold if high_threshold is None else high_threshold
    low_threshold = min(low_threshold, high_threshold)
    candidate = prob_np >= low_threshold
    seed = prob_np >= high_threshold
    height, width = prob_np.shape
    min_expand_area = max(1, int(round(height * width * min_expand_area_ratio)))
    if np.any(seed):
        refined = _expand_seeded_components(candidate, seed, min_expand_area)
    else:
        refined = raw_mask

    kernel_size = _odd_kernel_size(min(height, width), close_kernel_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    refined = cv2.morphologyEx(refined.astype("uint8"), cv2.MORPH_CLOSE, kernel).astype(bool)

    if fill_holes:
        refined = ndimage.binary_fill_holes(refined)

    min_component_area = max(1, int(round(height * width * min_component_area_ratio)))
    refined = _remove_small_components(refined, min_component_area, preserve=raw_mask)
    refined |= raw_mask
    return refined.astype("uint8") * 255


def estimate_content_mask(original, min_value=18, min_saturation=8, close_kernel_ratio=0.02):
    image = np.asarray(original.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    content = (value > min_value) | (saturation > min_saturation)

    height, width = content.shape
    kernel_size = _odd_kernel_size(min(height, width), close_kernel_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    content = cv2.morphologyEx(content.astype("uint8"), cv2.MORPH_CLOSE, kernel).astype(bool)
    content = ndimage.binary_fill_holes(content)

    min_area = max(1, int(round(height * width * 0.01)))
    content = _remove_small_components(content, min_area)
    return content.astype("uint8") * 255
