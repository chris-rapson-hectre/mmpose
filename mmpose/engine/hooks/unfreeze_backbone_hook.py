from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmpose.registry import HOOKS


@HOOKS.register_module()
class UnfreezeBackboneHook(Hook):
    def __init__(self, unfreeze_epoch, backbone_stages: int = 4):
        self.unfreeze_epoch = unfreeze_epoch
        self.backbone_stages = backbone_stages

    def before_run(self, runner: Runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if runner.epoch >= self.unfreeze_epoch:
            for param in model.backbone.parameters():
                param.requires_grad = True
            model.backbone.frozen_stages = -1
            model.backbone.train()
        else:
            model.backbone.frozen_stages = self.backbone_stages
            model.backbone._freeze_stages()
            runner.logger.info('Backbone frozen for initial training phase')

    def before_train_epoch(self, runner: Runner):
        model = runner.model
        if hasattr(model, 'module'):  # Handle DistributedDataParallel wrapper
            model = model.module
            
        # Check if we reached the target epoch    
        if runner.epoch >= self.unfreeze_epoch and model.backbone.frozen_stages != -1:
            runner.logger.info(f'Unfreezing backbone at epoch {runner.epoch}')


            # Set all backbone parameters to require gradients
            for param in model.backbone.parameters():
                param.requires_grad = True

            # Reset frozen_stages so HRNet.train() won't re-freeze
            model.backbone.frozen_stages = -1

            # Re-enable grad + switch to train mode (triggers _freeze_stages with -1)
            model.backbone.train()
