from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import torch

from mmpose.engine import UnfreezeBackboneHook


class DummyBackbone(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.frozen_stages = -1
        self.freeze_called = 0
        self.train_called = 0

    def _freeze_stages(self):
        self.freeze_called += 1
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        self.train_called += 1
        return super().train(mode)


class TestUnfreezeBackboneHook(TestCase):

    def test_before_run_freezes_backbone(self):
        """Verify backbone is frozen at initialization with correct frozen_stages."""
        hook = UnfreezeBackboneHook(unfreeze_epoch=3)

        model = SimpleNamespace(backbone=DummyBackbone())
        runner = SimpleNamespace(model=model, logger=Mock(), epoch=0)

        # confirm that model is not frozen until `before_run` is called
        assert model.backbone.frozen_stages == -1

        hook.before_run(runner)

        assert model.backbone.frozen_stages == 4
        assert model.backbone.freeze_called == 1
        assert all(
            not parameter.requires_grad
            for parameter in model.backbone.parameters()
        )

    def test_before_train_epoch_unfreezes_on_target_epoch_with_module_wrap(self):
        """Verify backbone unfreezes at target epoch, handling DistributedDataParallel wrapper."""
        hook = UnfreezeBackboneHook(unfreeze_epoch=2)

        wrapped_model = SimpleNamespace(backbone=DummyBackbone())
        runner = SimpleNamespace(
            model=SimpleNamespace(module=wrapped_model), logger=Mock(), epoch=2
        )

        hook.before_run(SimpleNamespace(model=wrapped_model, logger=Mock()))
        hook.before_train_epoch(runner)

        assert wrapped_model.backbone.frozen_stages == -1
        assert wrapped_model.backbone.train_called == 1
        assert all(
            parameter.requires_grad
            for parameter in wrapped_model.backbone.parameters()
        )

    def test_before_train_epoch_noop_when_not_target_epoch(self):
        """Verify backbone remains frozen when epoch does not match unfreeze target."""
        hook = UnfreezeBackboneHook(unfreeze_epoch=5)

        model = SimpleNamespace(backbone=DummyBackbone())
        runner = SimpleNamespace(model=model, logger=Mock(), epoch=2)

        hook.before_run(runner)
        hook.before_train_epoch(runner)

        assert model.backbone.frozen_stages == 4
        assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
        assert model.backbone.train_called == 0
