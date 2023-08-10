import hydra
import torch

from torch.nn import Module

class HumorLosses(Module):
    def __init__(self, vae: bool,
                 mode: str,
                 loss_on_both: bool = False,
                 force_loss_on_jfeats: bool = True,
                 ablation_no_kl_combine: bool = False,
                 ablation_no_motionencoder: bool = False,
                 ablation_no_kl_gaussian: bool = False, **kwargs):
        super().__init__()

        # Save parameters
        self.vae = vae
        self.mode = mode

        self.loss_on_both = loss_on_both
        self.force_loss_on_jfeats = force_loss_on_jfeats
        self.ablation_no_kl_combine = ablation_no_kl_combine
        self.ablation_no_kl_gaussian = ablation_no_kl_gaussian
        self.ablation_no_motionencoder = ablation_no_motionencoder

        losses = []
        
        if mode == "smpl":
            if not ablation_no_motionencoder:
                losses.append("recons_prior2rfeats")
            losses.append("recons_post2rfeats")
        else:
            ValueError("This mode is not recognized.")

        if vae or loss_on_both:
            kl_losses = []
            if not ablation_no_kl_combine and not ablation_no_motionencoder:
                kl_losses.extend(["kl_post2prior", "kl_prior2post"])
            if not ablation_no_kl_gaussian:
                if ablation_no_motionencoder:
                    kl_losses.extend(["kl_post"])
                else:
                    kl_losses.extend(["kl_post", "kl_prior"])
            losses.extend(kl_losses)
        if not self.vae or loss_on_both:
            if not ablation_no_motionencoder:
                losses.append("latent_manifold")
        losses.append("total")

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

    def update(self, ds_post=None, ds_prior=None, ds_ref=None,
               lat_post=None, lat_prior=None, dis_post=None,
               dis_prior=None, dis_ref=None):
        total: float = 0.0

        if self.mode == "smpl":
            if not self.ablation_no_motionencoder:
                total += self._update_loss("recons_post2rfeats", ds_post.rfeats, ds_ref.rfeats)
            total += self._update_loss("recons_prior2rfeats", ds_prior.rfeats, ds_ref.rfeats)

        if self.vae or self.loss_on_both:
            if not self.ablation_no_kl_combine and not self.ablation_no_motionencoder:
                total += self._update_loss("kl_post2prior", dis_post, dis_prior)
                total += self._update_loss("kl_prior2post", dis_prior, dis_post)
            if not self.ablation_no_kl_gaussian:
                total += self._update_loss("kl_post", dis_post, dis_ref)
                if not self.ablation_no_motionencoder:
                    total += self._update_loss("kl_prior", dis_prior, dis_ref)
        if not self.vae or self.loss_on_both:
            if not self.ablation_no_motionencoder:
                total += self._update_loss("latent_manifold", lat_post, lat_prior)

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
