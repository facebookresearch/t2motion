import hydra
import torch

from torch.nn import Module

class CLFTemosComputeLosses(Module):
    def __init__(self, **kwargs):
        super().__init__()

        losses = []
        losses.append("total")
        losses.append("clf_motion")

        self.losses_values = {}
        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))

        self.register_buffer("count", torch.tensor(0.0))
        self.losses = losses

        # Instantiate loss functions
        self._losses_func = {loss: hydra.utils.instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss != "total"}
        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def update(self, motion_probs=None, gt_labels=None):
        total: float = 0.0
        total += self._update_loss("clf_motion", motion_probs, gt_labels)

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = self.count
        # return {loss: self.losses_values[loss]/count for loss in self.losses}
        return {loss: getattr(self, loss)/count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        # self.losses_values[loss] += val.detach()
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
