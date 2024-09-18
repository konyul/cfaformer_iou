from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.optimizer import Fp16OptimizerHook


@HOOKS.register_module()
class CustomFp16OptimizerHook(Fp16OptimizerHook):

    def __init__(self,
                 custom_fp16={},
                 cumulative_iters=None,
                 *args,
                 **kwargs):
        super(CustomFp16OptimizerHook, self).__init__(*args, **kwargs)
        # self.cumulative_iters = cumulative_iters
        # self._inner_count = 0
        self.custom_fp16 = custom_fp16

    def before_run(self, runner) -> None:
        super().before_run(runner)
        for module_name, v in self.custom_fp16.items():
            runner.model.module._modules[module_name].fp16_enabled = v
    # def after_train_iter(self, runner) -> None:
    #         """Backward optimization steps for Mixed Precision Training. For
    #         dynamic loss scaling, please refer to
    #         https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

    #         1. Scale the loss by a scale factor.
    #         2. Backward the loss to obtain the gradients.
    #         3. Unscale the optimizer�~@~Ys gradient tensors.
    #         4. Call optimizer.step() and update scale factor.
    #         5. Save loss_scaler state_dict for resume purpose.
    #         """
    #         # clear grads of last iteration
    #         runner.model.zero_grad()
    #         if ((self._inner_count+1) % self.cumulative_iters==0):
    #             runner.optimizer.zero_grad()

    #         self.loss_scaler.scale(runner.outputs['loss']).backward()
    #         self.loss_scaler.unscale_(runner.optimizer)
    #         # grad clip
    #         if self.grad_clip is not None:
    #             grad_norm = self.clip_grads(runner.model.parameters())
    #             if grad_norm is not None:
    #                 # Add grad norm to the logger
    #                 runner.log_buffer.update({'grad_norm': float(grad_norm)},
    #                                          runner.outputs['num_samples'])
    #         # backward and update scaler
    #         if ((self._inner_count+1) % self.cumulative_iters==0):
    #             self.loss_scaler.step(runner.optimizer)
    #         self.loss_scaler.update(self._scale_update_param)
    #         self._inner_count += 1

    #         # save state_dict of loss_scaler
    #         runner.meta.setdefault(
    #             'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()
