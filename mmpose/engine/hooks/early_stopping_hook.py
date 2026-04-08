from typing import Optional

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmpose.registry import HOOKS


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Stop training when a monitored metric stops improving.

    After each validation run, checks whether the monitored metric has
    improved compared to the best value seen so far.  The patience counter
    increments only when the metric fails to move in the right direction at
    all.  ``min_delta`` controls what counts as a meaningful improvement for
    updating the *best score*, but any positive movement still resets the
    counter — matching the intent of "allow for noise when the rate of
    improvement is still small but in the right direction".

    Args:
        monitor (str): Metric key to watch, e.g. ``'coco/AP'``.
        patience (int): Number of consecutive validation checks with no
            improvement before training is stopped.  Defaults to 10.
        min_delta (float): Minimum change to update the recorded best score.
            Movements smaller than this do *not* update the best, but they
            *do* reset the patience counter as long as they are in the right
            direction.  Defaults to 0.0.
        rule (str): ``'greater'`` — higher is better (e.g. AP, accuracy);
                    ``'less'``    — lower is better (e.g. loss, NME).
            Defaults to ``'greater'``.
    """

    _init_values = {'greater': -float('inf'), 'less': float('inf')}

    def __init__(
        self,
        monitor: str,
        patience: int = 10,
        min_delta: float = 0.0,
        rule: str = 'greater',
    ) -> None:
        assert rule in ('greater', 'less'), \
            f'rule must be "greater" or "less", got "{rule}"'
        assert patience >= 1, 'patience must be >= 1'
        assert min_delta >= 0.0, 'min_delta must be >= 0'

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.rule = rule
        self.best_score = self._init_values[rule]
        self.wait_count = 0

    def _is_moving_right_direction(self, current: float) -> bool:
        """Any improvement at all — resets patience."""
        if self.rule == 'greater':
            return current > self.best_score
        return current < self.best_score

    def _is_meaningful_improvement(self, current: float) -> bool:
        """Improvement large enough to update the recorded best score."""
        if self.rule == 'greater':
            return current > self.best_score + self.min_delta
        return current < self.best_score - self.min_delta

    def after_val_epoch(self, runner: Runner, metrics: Optional[dict] = None) -> None:
        if not metrics:
            return

        if self.monitor not in metrics:
            runner.logger.warning(
                f'EarlyStoppingHook: "{self.monitor}" not found in metrics '
                f'{sorted(metrics.keys())}. Skipping check.')
            return

        current = float(metrics[self.monitor])

        if self._is_meaningful_improvement(current):
            # Genuine improvement: update best and reset counter.
            runner.logger.info(
                f'EarlyStoppingHook: {self.monitor} improved '
                f'{self.best_score:.4f} → {current:.4f}. '
                f'Patience counter reset.')
            self.best_score = current
            self.wait_count = 0

        elif self._is_moving_right_direction(current):
            # Small improvement (< min_delta): reset counter but don't
            # update best_score so the bar keeps rising.
            runner.logger.info(
                f'EarlyStoppingHook: {self.monitor} = {current:.4f} — '
                f'small improvement over best {self.best_score:.4f} '
                f'(< min_delta {self.min_delta:.4f}). '
                f'Patience counter reset.')
            self.wait_count = 0

        else:
            # No improvement or degradation.
            self.wait_count += 1
            runner.logger.info(
                f'EarlyStoppingHook: {self.monitor} = {current:.4f} '
                f'(best = {self.best_score:.4f}). '
                f'No improvement: {self.wait_count}/{self.patience}.')

            if self.wait_count >= self.patience:
                runner.logger.info(
                    f'EarlyStoppingHook: stopping training — '
                    f'{self.monitor} has not improved for {self.patience} '
                    f'consecutive validation checks '
                    f'(best = {self.best_score:.4f}).')
                self._stop(runner)

    @staticmethod
    def _stop(runner: Runner) -> None:
        """Signal the EpochBasedTrainLoop to break after the current epoch."""
        if hasattr(runner.train_loop, 'stop_training'):
            # Preferred: MMEngine sets this flag at the end of each epoch
            # iteration and breaks the loop cleanly.
            runner.train_loop.stop_training = True
        else:
            # Older MMEngine fallback — won't affect an already-running
            # range() but will at least prevent resuming.
            runner.train_loop._max_epochs = runner.epoch
            runner.logger.warning(
                'EarlyStoppingHook: train_loop.stop_training not available. '
                'Consider upgrading MMEngine.')