# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from mmpose.engine.hooks import EarlyStoppingHook


class TestEarlyStoppingHook(TestCase):

    def test_validation_on_init(self):
        """Verify constructor validates rule, patience, and min_delta parameters."""
        with self.assertRaises(AssertionError):
            EarlyStoppingHook(monitor="metric", rule="invalid")
        with self.assertRaises(AssertionError):
            EarlyStoppingHook(monitor="metric", patience=0)
        with self.assertRaises(AssertionError):
            EarlyStoppingHook(monitor="metric", min_delta=-0.1)

    def test_greater_rule_with_small_improvement_and_stop(self):
        """Verify 'greater' rule: meaningful/small improvements reset patience, no improvement triggers stop."""
        hook = EarlyStoppingHook(
            monitor="coco/AP", patience=2, min_delta=0.1, rule="greater"
        )

        runner = SimpleNamespace(
            logger=Mock(), train_loop=SimpleNamespace(stop_training=False), epoch=9
        )

        hook.after_val_epoch(runner, {"coco/AP": 0.5})
        assert hook.best_score == 0.5
        assert hook.wait_count == 0
        assert runner.train_loop.stop_training == False

        hook.after_val_epoch(runner, {"coco/AP": 0.55})
        assert hook.best_score == 0.5
        assert hook.wait_count == 0

        hook.after_val_epoch(runner, {"coco/AP": 0.49})
        assert hook.wait_count == 1
        assert runner.train_loop.stop_training == False

        hook.after_val_epoch(runner, {"coco/AP": 0.48})
        assert hook.wait_count == 2
        assert runner.train_loop.stop_training == True

    def test_less_rule_improves_when_metric_decreases(self):
        """Verify 'less' rule: metric decrease is treated as improvement and updates best score."""
        hook = EarlyStoppingHook(monitor="nme", patience=2, min_delta=0.01, rule="less")
        runner = SimpleNamespace(
            logger=Mock(), train_loop=SimpleNamespace(stop_training=False), epoch=3
        )

        hook.after_val_epoch(runner, {"nme": 0.2})
        hook.after_val_epoch(runner, {"nme": 0.189})  # > 0.01 (min-delta)

        assert hook.best_score == 0.189
        assert hook.wait_count == 0

    def test_missing_metric_is_skipped(self):
        """Verify missing monitored metric logs warning and does not update state."""
        hook = EarlyStoppingHook(monitor="missing_key")
        runner = SimpleNamespace(
            logger=Mock(), train_loop=SimpleNamespace(stop_training=False), epoch=1
        )

        hook.after_val_epoch(runner, {"other": 1.0})

        assert hook.best_score == -float("inf")
        assert hook.wait_count == 0
        runner.logger.warning.assert_called_once()

    def test_stop_fallback_without_stop_training_flag(self):
        """Verify fallback stop mechanism when stop_training flag is not available."""
        runner = SimpleNamespace(
            logger=Mock(), train_loop=SimpleNamespace(_max_epochs=100), epoch=7
        )

        EarlyStoppingHook._stop(runner)

        assert runner.train_loop._max_epochs == 7
        runner.logger.warning.assert_called_once()

