from unittest import TestCase

import numpy as np

from mmpose.codecs.utils.refinement import refine_keypoints_dark_udp


class TestRefineKeypointsDarkUDP(TestCase):

    def test_flat_heatmap_keeps_keypoints_unchanged(self):
        """
        This test has two goals:
        1. Verify that the change from `inv` to `pinv` means the function won't fail with a singular Hessian,
        e.g., when the heatmap is flat.
        2. Serve as a baseline for the next test.
        """
        heatmaps = np.ones((1, 21, 21), dtype=np.float32)
        keypoints = np.array([[[10.0, 10.0]]], dtype=np.float32)

        refined = refine_keypoints_dark_udp(keypoints.copy(), heatmaps.copy(), blur_kernel_size=11)

        assert np.all(np.isfinite(refined))
        np.testing.assert_allclose(refined, keypoints)

    def test_gaussian_heatmap_moves_towards_true_peak(self):
        height = width = 21
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        true_peak = np.array([10.25, 9.75], dtype=np.float32)
        sigma = 2.0

        heatmaps = np.exp(
            -((x_grid - true_peak[0]) ** 2 + (y_grid - true_peak[1]) ** 2)
            / (2 * sigma**2)
        ).astype(np.float32)[None, ...]
        keypoints = np.array([[[10.0, 10.0]]], dtype=np.float32)

        refined = refine_keypoints_dark_udp(keypoints.copy(), heatmaps.copy(), blur_kernel_size=11)

        refined_xy = refined[0, 0, :2]
        initial_xy = keypoints[0, 0, :2]

        assert refined_xy[0] > initial_xy[0]
        assert refined_xy[1] < initial_xy[1]
        # Euclidean distance to true peak should be smaller after refinement
        assert np.linalg.norm(refined_xy - true_peak) < np.linalg.norm(initial_xy - true_peak)
