import os

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DebugVisualizeAugmented(BaseTransform):
    """Save augmented training images with keypoints and skeleton overlaid.

    Insert this into the pipeline after all spatial and pixel-level
    augmentations, but before GenerateTarget / PackPoseInputs.

    Args:
        out_dir (str): Directory to save debug images.
        max_samples (int): Stop saving after this many images.
            Set to -1 for unlimited. Defaults to 200.
        kpt_radius (int): Radius of keypoint circles. Defaults to 4.
        link_thickness (int): Thickness of skeleton lines. Defaults to 2.
    """

    _counter = 0  # class-level counter shared across workers

    def __init__(self,
                 out_dir: str = 'debug_augmented',
                 max_samples: int = 200,
                 kpt_radius: int = 4,
                 link_thickness: int = 2) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.max_samples = max_samples
        self.kpt_radius = kpt_radius
        self.link_thickness = link_thickness
        os.makedirs(out_dir, exist_ok=True)

    def transform(self, results: dict) -> dict:
        if 0 <= self.max_samples <= DebugVisualizeAugmented._counter:
            return results

        img = results['img'].copy()
        # Convert RGB to BGR for cv2 if needed (LoadImage loads as RGB)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        keypoints = results.get('keypoints', None)
        keypoints_visible = results.get('keypoints_visible', None)
        skeleton_links = results.get('skeleton_links', None)

        if keypoints is not None:
            # keypoints shape: (N, K, 2) for bottomup
            for inst_idx, (kpts, vis) in enumerate(
                    zip(keypoints, keypoints_visible)):
                # Draw skeleton links first (so dots are on top)
                if skeleton_links is not None:
                    for link in skeleton_links:
                        k1, k2 = link
                        if vis[k1] > 0 and vis[k2] > 0:
                            pt1 = tuple(kpts[k1].astype(int))
                            pt2 = tuple(kpts[k2].astype(int))
                            cv2.line(img, pt1, pt2, (200, 200, 200),
                                     self.link_thickness, cv2.LINE_AA)

                # Draw keypoints with index labels
                for k_idx, (pt, v) in enumerate(zip(kpts, vis)):
                    if v <= 0:
                        continue
                    x, y = int(pt[0]), int(pt[1])
                    # Colour cycles through hue so each keypoint is distinct
                    hue = int(k_idx * 180 / max(len(kpts), 1))
                    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                    color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

                    cv2.circle(img, (x, y), self.kpt_radius, color, -1,
                               cv2.LINE_AA)
                    cv2.circle(img, (x, y), self.kpt_radius, (0, 0, 0), 1,
                               cv2.LINE_AA)
                    cv2.putText(img, str(k_idx), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (255, 255, 255), 1, cv2.LINE_AA)

        # Build filename from source image name + counter
        img_path = results.get('img_path', '')
        basename = os.path.splitext(os.path.basename(img_path))[0]
        fname = f'{DebugVisualizeAugmented._counter:06d}_{basename}.jpg'

        # img is RGB from the pipeline, convert to BGR for cv2.imwrite
        cv2.imwrite(os.path.join(self.out_dir, fname),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        DebugVisualizeAugmented._counter += 1

        return results