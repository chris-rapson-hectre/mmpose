from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from mmpose.datasets import RandomRot90, RandomFlipBidirectional
from mmpose.testing import get_coco_sample


# Mappings for a quadrilateral with one point on each corner and two points along each side
# keypoints: [0, tl, 2, 3, tr, 5, 6, br, 8, 9, bl, 11]
TL, TR, BR, BL = 1, 4, 7, 10
GREEN = np.array([0, 255, 0], dtype=np.uint8)
MARKER_COLOR = np.array([255, 0, 0], dtype=np.uint8)


# new_index <- old_index after k=1 (90deg CCW)
ROT90_INDICES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]

# swap pairs
FLIP_INDICES = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]
FLIP_UD_INDICES = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# x1y1x2y2 format
EXPECTED_SYMMETRIC_BBOX = {
    "original": np.array([[1, 1, 10, 6]], dtype=np.float32),
    "rotated": np.array([[1, 1, 6, 10]], dtype=np.float32),
}
EXPECTED_ASYMMETRIC_BBOX = {
    "original": np.array([[0, 0, 10, 6]], dtype=np.float32),
    "rotated90": np.array([[0, 1, 6, 11]], dtype=np.float32),
    "flipped_horizontal": np.array([[1, 0, 11, 6]], dtype=np.float32),
    "flipped_vertical": np.array([[0, 1, 10, 7]], dtype=np.float32),
    "flipped_both": np.array([[1, 1, 11, 7]], dtype=np.float32),
}

EXPECTED_KEYPOINTS = {
    "original": np.array(
        [
            [
                [1, 2],
                [1, 1],
                [2, 1],
                [9, 1],
                [10, 1],
                [10, 2],
                [10, 5],
                [10, 6],
                [9, 6],
                [2, 6],
                [1, 6],
                [1, 5],
            ]
        ],
        dtype=np.float32,
    ),
    "rotated": np.array(
        [
            [
                [1, 2],
                [1, 1],
                [2, 1],
                [5, 1],
                [6, 1],
                [6, 2],
                [6, 9],
                [6, 10],
                [5, 10],
                [2, 10],
                [1, 10],
                [1, 9],
            ]
        ],
        dtype=np.float32,
    ),
}


EXPECTED_ASSYMMETRIC_KEYPOINTS = {
    "original": np.array(
        [
            [
                [0, 1],
                [0, 0],  # pulled out corner
                [1, 0],
                [9, 1],
                [10, 1],  # normal corner
                [10, 2],
                [9, 4],
                [9, 5],  # pushed in corner
                [8, 5],
                [2, 6],
                [1, 6],  # normal corner
                [1, 5],
            ]
        ],
    ),
    "rotated90": np.array(
        [
            [
                [1, 2],
                [1, 1],  # normal corner
                [2, 1],
                [4, 2],
                [5, 2],  # pushed in corner
                [5, 3],
                [6, 9],
                [6, 10],  # normal corner
                [5, 10],
                [1, 11],
                [0, 11],  # pulled out corner
                [0, 10],
            ]
        ],
    ),
    "flipped_horizontal": np.array(
        [
            [
                [1, 2],
                [1, 1],  # normal corner
                [2, 1],
                [10, 0],
                [11, 0],  # pulled out corner
                [11, 1],
                [10, 5],
                [10, 6],  # normal corner
                [9, 6],
                [3, 5],
                [2, 5],  # pushed in corner
                [2, 4],
            ]
        ],
    ),
    "flipped_vertical": np.array(
        [
            [
                [1, 2],
                [1, 1],  # normal corner
                [2, 1],
                [8, 2],
                [9, 2],  # pushed in
                [9, 3],
                [10, 5],
                [10, 6],  # normal corner
                [9, 6],
                [1, 7],
                [0, 7],  # pulled out corner
                [0, 6],
            ]
        ]
    ),
    "flipped_both": np.array(
        [
            [
                [2, 3],
                [2, 2],  # pushed in corner
                [3, 2],
                [9, 1],
                [10, 1],  # normal corner
                [10, 2],
                [11, 6],
                [11, 7],  # pulled out corner
                [10, 7],
                [2, 6],
                [1, 6],  # normal corner
                [1, 5],
            ]
        ]
    ),
}


def _expected_corners_from_keypoints(
    expected_keypoints: np.ndarray,
) -> list[tuple[int, int]]:
    return [
        tuple(expected_keypoints[0][TL]),
        tuple(expected_keypoints[0][TR]),
        tuple(expected_keypoints[0][BR]),
        tuple(expected_keypoints[0][BL]),
    ]


def _assert_color_pixel_count(
    image: np.ndarray,
    color: np.ndarray,
    expected_count: int
) -> None:
    count = np.sum(np.all(image == color, axis=-1))
    np.testing.assert_equal(count, expected_count)


def _setup_test_image_and_keypoints() -> dict:
    h, w = 8, 12
    results = get_coco_sample(img_shape=(h, w), num_instances=1, with_bbox_cs=False)

    results["img"] = np.zeros((h, w, 3), dtype=np.uint8)
    results["img"][0, w - 1] = MARKER_COLOR

    # xy corner coordinates
    corner_coordinates = {
        TL: [1, 1],
        TR: [w - 2, 1],
        BR: [w - 2, h - 2],
        BL: [1, h - 2],
    }

    for _, xy in corner_coordinates.items():
        results["img"][xy[1], xy[0]] = GREEN

    # x1y1x2y2 format
    results["bbox"] = EXPECTED_SYMMETRIC_BBOX["original"]
    results["bbox_scale"] = np.array([results["bbox"][0][2] - results["bbox"][0][0], results["bbox"][0][3] - results["bbox"][0][1]])

    # Corner locations before rotation (clockwise order on image).
    keypoints = np.zeros((1, 12, 2), dtype=np.float32)
    keypoints[0, TL] = corner_coordinates[TL]
    keypoints[0, TR] = corner_coordinates[TR]
    keypoints[0, BR] = corner_coordinates[BR]
    keypoints[0, BL] = corner_coordinates[BL]

    # Fill the remaining ids with unique points (xy)
    keypoints[0, 0] = [1, 2]  # below TL
    keypoints[0, 2] = [2, 1]  # right of TL
    keypoints[0, 3] = [w - 3, 1]  # left of TR
    keypoints[0, 5] = [w - 2, 2]  # below TR
    keypoints[0, 6] = [w - 2, h - 3]  # above BR
    keypoints[0, 8] = [w - 3, h - 2]  # left of BR
    keypoints[0, 9] = [2, h - 2]  # right of BL
    keypoints[0, 11] = [1, h - 3]  # above BL

    keypoints_visible = np.arange(12, dtype=np.float32)[None, :]

    results["keypoints"] = keypoints
    results["keypoints_visible"] = keypoints_visible

    return {
        "h": h,
        "w": w,
        "results": results,
        "corner_coordinates": corner_coordinates,
    }


def _setup_asymmetric_image_and_keypoints() -> dict:
    # Create an asymmetric image and keypoints to test flipping and rotation
    h, w = 8, 12
    results = get_coco_sample(img_shape=(h, w), num_instances=1, with_bbox_cs=False)

    results["img"] = np.zeros((h, w, 3), dtype=np.uint8)
    results["img"][0, w - 1] = MARKER_COLOR

    # Asymmetric corner coordinates (not centered) xy format
    corner_coordinates = {
        TL: [0, 0],  # pulled out
        TR: [w - 2, 1],
        BR: [w - 3, h - 3],  # pushed in
        BL: [1, h - 2],
    }

    for _, xy in corner_coordinates.items():
        results["img"][xy[1], xy[0]] = GREEN

    #     # x1y1x2y2 format
    results["bbox"] = EXPECTED_ASYMMETRIC_BBOX["original"]
    results["bbox_scale"] = np.array([results["bbox"][0][2] - results["bbox"][0][0], results["bbox"][0][3] - results["bbox"][0][1]])

    # Corner locations before rotation (clockwise order on image).
    keypoints = np.zeros((1, 12, 2), dtype=np.float32)
    keypoints[0, TL] = corner_coordinates[TL]
    keypoints[0, TR] = corner_coordinates[TR]
    keypoints[0, BR] = corner_coordinates[BR]
    keypoints[0, BL] = corner_coordinates[BL]

    # Fill the remaining ids with unique points (xy)
    keypoints[0, 0] = [0, 1]  # below TL
    keypoints[0, 2] = [1, 0]  # right of TL
    keypoints[0, 3] = [w - 3, 1]  # left of TR
    keypoints[0, 5] = [w - 2, 2]  # below TR
    keypoints[0, 6] = [w - 3, h - 4]  # above BR
    keypoints[0, 8] = [w - 4, h - 3]  # left of BR
    keypoints[0, 9] = [2, h - 2]  # right of BL
    keypoints[0, 11] = [1, h - 3]  # above BL

    keypoints_visible = np.arange(12, dtype=np.float32)[None, :]

    results["keypoints"] = keypoints
    results["keypoints_visible"] = keypoints_visible

    return {
        "h": h,
        "w": w,
        "results": results,
        "corner_coordinates": corner_coordinates,
    }


class TestRandomRot90(TestCase):

    def setUp(self):
        data = _setup_test_image_and_keypoints()
        self.h = data["h"]
        self.w = data["w"]
        self.results = data["results"]
        self.corner_coordinates = data["corner_coordinates"]

        self.results["rot90_indices"] = ROT90_INDICES
        self.transform = RandomRot90(prob=1.0)

    def _run_transform(self, results: dict, k: int) -> dict:
        with patch.object(self.transform, "_get_rot_k", return_value=k):
            return self.transform.transform(deepcopy(results))

    def test_rotate_ccw_updates_image_shape_and_marker(self):
        for k in range(4):
            with self.subTest(k=k):
                out = self._run_transform(self.results, k)

            assert out is not None
            if k % 2 == 0:  # 180 or 360 degree rotation keeps shape
                np.testing.assert_array_equal(out["img_shape"], (self.h, self.w))
            else:  # 90 or 270 degree rotation swaps height and width
                np.testing.assert_array_equal(out["img_shape"], (self.w, self.h))

            # assert the marker is in the expected corner after rotation
            if k == 0:  # 0 degree, marker at top-right
                np.testing.assert_array_equal(out["img"][0, self.w - 1], MARKER_COLOR)
            elif k == 1:  # 90 degree CCW, marker at top-left
                np.testing.assert_array_equal(out["img"][0, 0], MARKER_COLOR)
            elif k == 2:  # 180 degree, marker at bottom-left
                np.testing.assert_array_equal(out["img"][self.h - 1, 0], MARKER_COLOR)
            elif k == 3:  # 270 degree CCW, marker at bottom-right
                # swapped index as swap of height and width
                np.testing.assert_array_equal(out["img"][self.w - 1, self.h - 1], MARKER_COLOR)

            # Assert only one pixel is the marker, and the rest are not
            marker_count = np.sum(np.all(out["img"] == MARKER_COLOR, axis=-1))
            np.testing.assert_equal(marker_count, 1)

    def test_rotate_ccw_moves_corner_pixels_to_expected_locations(self):
        for k in range(4):
            with self.subTest(k=k):
                out = self._run_transform(self.results, k)

                for _, xy in self.corner_coordinates.items():
                    if k % 2 == 1:  # odd k swaps x and y
                        xy = [xy[1], xy[0]]
                    np.testing.assert_array_equal(out["img"][xy[1], xy[0]], GREEN)

                # Assert only the 4 corner pixels are green, and the rest are not
                green_count = np.sum(np.all(out["img"] == GREEN, axis=-1))
                np.testing.assert_equal(green_count, 4)

    def test_rotate_ccw_symmetric_centred_bbox_rotates(self):
        for k in range(4):
            with self.subTest(k=k):
                out = self._run_transform(self.results, k)
                new_bbox = out["bbox"]  # [x1, y1, x2, y2]
                new_bbox_scale = out["bbox_scale"]

                if k % 2 == 1:  # odd k swaps width and height
                    np.testing.assert_array_equal(new_bbox, EXPECTED_SYMMETRIC_BBOX["rotated"])
                    np.testing.assert_array_equal(new_bbox_scale, self.results["bbox_scale"][::-1])
                else:
                    np.testing.assert_array_equal(new_bbox, EXPECTED_SYMMETRIC_BBOX["original"])
                    np.testing.assert_array_equal(new_bbox_scale, self.results["bbox_scale"])

    def test_rotate_ccw_applies_exact_keypoint_mapping(self):
        for k in range(4):
            with self.subTest(k=k):
                # The transform re-orders keypoints according to ROT90_INDICES
                out = self._run_transform(self.results, k)
                if k % 2 == 0:
                    np.testing.assert_array_equal(out["keypoints"], EXPECTED_KEYPOINTS["original"])
                else:
                    np.testing.assert_array_equal(out["keypoints"], EXPECTED_KEYPOINTS["rotated"])

    def test_rotate_asymmetric_image_and_keypoints(self):
        data = _setup_asymmetric_image_and_keypoints()
        h = data["h"]
        w = data["w"]
        results = data["results"]

        results["rot90_indices"] = ROT90_INDICES

        # get corners from expected keypoints after rotation, to verify pixel values in output image
        expected_rotated_corners = _expected_corners_from_keypoints(EXPECTED_ASSYMMETRIC_KEYPOINTS["rotated90"])

        out = self._run_transform(results, k=1)

        np.testing.assert_array_equal(out["img_shape"], (w, h))
        np.testing.assert_array_equal(out["img"][0, 0], MARKER_COLOR)  # marker

        for xy in expected_rotated_corners:
            np.testing.assert_array_equal(out["img"][xy[1], xy[0]], GREEN)  # corners


        np.testing.assert_array_equal(out["bbox"], EXPECTED_ASYMMETRIC_BBOX["rotated90"])  # bbox
        np.testing.assert_array_equal(out["bbox_scale"], results["bbox_scale"][::-1])

        np.testing.assert_array_equal(out["keypoints"], EXPECTED_ASSYMMETRIC_KEYPOINTS["rotated90"])  # keypoints


class TestRandomFlipBidirectional(TestCase):

    def setUp(self):
        data = _setup_asymmetric_image_and_keypoints()
        self.h = data["h"]
        self.w = data["w"]
        self.results = data["results"]

        self.results["flip_indices"] = FLIP_INDICES
        self.results["flip_ud_indices"] = FLIP_UD_INDICES

    def _run_transform(self, directions: list[str]) -> dict:
        transform = RandomFlipBidirectional(prob=1.0, directions=directions)
        return transform.transform(deepcopy(self.results))

    def _assert_flip_output(
        self,
        out: dict,
        marker_xy: tuple[int, int],
        expected_key: str,
        expected_flip: bool,
    ) -> None:
        expected_keypoints = EXPECTED_ASSYMMETRIC_KEYPOINTS[expected_key]
        expected_corners = _expected_corners_from_keypoints(expected_keypoints)

        np.testing.assert_array_equal(out["img_shape"], (self.h, self.w))
        np.testing.assert_array_equal(out["img"][marker_xy[1], marker_xy[0]], MARKER_COLOR)  # marker
        _assert_color_pixel_count(out["img"], MARKER_COLOR, 1)

        for xy in expected_corners:
            np.testing.assert_array_equal(out["img"][xy[1], xy[0]], GREEN)

        _assert_color_pixel_count(out["img"], GREEN, 4)

        np.testing.assert_array_equal(out["bbox"], EXPECTED_ASYMMETRIC_BBOX[expected_key])
        np.testing.assert_array_equal(out["keypoints"], expected_keypoints)

        assert out['flip'] == expected_flip

    def test_flip_horizontal(self):
        out = self._run_transform(["horizontal"])
        self._assert_flip_output(
            out=out,
            marker_xy=(0, 0),
            expected_key="flipped_horizontal",
            expected_flip=True,
        )

    def test_flip_vertical(self):
        out = self._run_transform(["vertical"])
        self._assert_flip_output(
            out=out,
            marker_xy=(self.w - 1, self.h - 1),
            expected_key="flipped_vertical",
            expected_flip=True,
        )

    def test_flip_both(self):
        out = self._run_transform(["horizontal", "vertical"])
        self._assert_flip_output(
            out=out,
            marker_xy=(0, self.h - 1),
            expected_key="flipped_both",
            expected_flip=True,
        )

    def test_flip_neither(self):
        out = self._run_transform([])
        self._assert_flip_output(
            out=out,
            marker_xy=(self.w - 1, 0),
            expected_key="original",
            expected_flip=False,
        )
