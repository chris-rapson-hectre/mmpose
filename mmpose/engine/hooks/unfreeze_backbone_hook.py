from mmengine.hooks import Hook

from mmpose.registry import HOOKS


@HOOKS.register_module()
class UnfreezeBackboneHook(Hook):
    def __init__(self, unfreeze_epoch):
        self.unfreeze_epoch = unfreeze_epoch

    def before_run(self, runner):
        # Firstly, freeze the backbone, then unfreeze later
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        model.backbone.frozen_stages = 4
        model.backbone._freeze_stages()
        runner.logger.info('Backbone frozen for initial training phase')

    def before_train_epoch(self, runner):
        # Check if we reached the target epoch
        if runner.epoch == self.unfreeze_epoch:
            runner.logger.info(f'Unfreezing backbone at epoch {runner.epoch}')
            model = runner.model
            if hasattr(model, 'module'):  # Handle DistributedDataParallel wrapper
                model = model.module

            # Set all backbone parameters to require gradients
            for param in model.backbone.parameters():
                param.requires_grad = True

            # Reset frozen_stages so HRNet.train() won't re-freeze
            model.backbone.frozen_stages = -1

            # Re-enable grad + switch to train mode (triggers _freeze_stages with -1)
            model.backbone.train()
