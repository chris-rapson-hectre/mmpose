# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from mmengine.structures import InstanceData

from mmpose.datasets import DebugVisualizeAugmented
from mmpose.engine import DebugAugmentedSetupHook

from mmpose.engine.hooks import PoseVisualizationHook
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


def _rand_poses(num_boxes, h, w):
    center = np.random.rand(num_boxes, 2)
    offset = np.random.rand(num_boxes, 5, 2) / 2.0

    pose = center[:, None, :] + offset.clip(0, 1)
    pose[:, :, 0] *= w
    pose[:, :, 1] *= h

    return pose


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        PoseLocalVisualizer.get_instance('test_visualization_hook')

        data_sample = PoseDataSample()
        data_sample.set_metainfo({
            'img_path':
            osp.join(
                osp.dirname(__file__), '../../data/coco/000000000785.jpg')
        })
        self.data_batch = {'data_samples': [data_sample] * 2}

        pred_instances = InstanceData()
        pred_instances.keypoints = _rand_poses(5, 10, 12)
        pred_instances.score = np.random.rand(5, 5)
        pred_det_data_sample = data_sample.clone()
        pred_det_data_sample.pred_instances = pred_instances
        self.outputs = [pred_det_data_sample] * 2

    def test_after_val_iter(self):
        runner = MagicMock()
        runner.iter = 1
        runner.val_evaluator.dataset_meta = dict()
        hook = PoseVisualizationHook(interval=1, enable=True)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = MagicMock()
        runner.iter = 1
        hook = PoseVisualizationHook(enable=True)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertEqual(hook._test_index, 2)

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = PoseVisualizationHook(enable=False, out_dir=out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{out_dir}'))

        hook = PoseVisualizationHook(enable=True, out_dir=out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{out_dir}'))
        shutil.rmtree(f'{timestamp}')


class TestDebugAugmentedSetupHook(TestCase):

    def setUp(self):
        self._old_env_value = os.environ.get("MMPOSE_DEBUG_AUG_DIR")

    def tearDown(self):
        if self._old_env_value is None:
            os.environ.pop("MMPOSE_DEBUG_AUG_DIR", None)
        else:
            os.environ["MMPOSE_DEBUG_AUG_DIR"] = self._old_env_value

    def test_before_run_sets_env_and_updates_transform(self):
        """Verify env var and transform output dir are set correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hook = DebugAugmentedSetupHook()
            transform = DebugVisualizeAugmented(out_dir="old-path")
            transform._resolved_out_dir = "already-resolved"
            pipeline = SimpleNamespace(transforms=[transform])

            runner = SimpleNamespace(
                work_dir=temp_dir,
                timestamp="20260330_120000",
                train_dataloader=SimpleNamespace(
                    dataset=SimpleNamespace(pipeline=pipeline)
                ),
            )

            hook.before_run(runner)

            expected_dir = os.path.join(temp_dir, runner.timestamp, "debug_augmented")
            assert os.environ.get("MMPOSE_DEBUG_AUG_DIR") == expected_dir
            assert transform._configured_out_dir == expected_dir
            assert transform._resolved_out_dir == None
            assert os.path.isdir(expected_dir)

    def test_before_run_sets_env_even_without_pipeline(self):
        """Verify env var is set even when pipeline attribute is unavailable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hook = DebugAugmentedSetupHook()
            runner = SimpleNamespace(
                work_dir=temp_dir,
                timestamp="20260330_120000",
                train_dataloader=SimpleNamespace(dataset=SimpleNamespace()),
            )

            hook.before_run(runner)

            expected_dir = os.path.join(temp_dir, runner.timestamp, "debug_augmented")
            assert os.environ.get("MMPOSE_DEBUG_AUG_DIR") == expected_dir
