import hydra
import torch

from torch.nn import Module
import torch.nn.functional as F

class EMSComputeLosses(Module):
    def __init__(self, vae: bool,
                 mode: str,
                 loss_on_both: bool = False,
                 force_loss_on_jfeats: bool = True,
                 ablation_no_kl_combine: bool = False,
                 ablation_no_motionencoder: bool = False,
                 discriminator: bool = False,
                 bigen: bool = False,
                 humor: bool = False,
                 fskate: bool = False,
                 contrast: bool = False,
                 ablation_no_kl_gaussian: bool = False, **kwargs):
        super().__init__()

        # Save parameters
        self.vae = vae
        self.mode = mode
        self.discriminator = discriminator
        self.bigen = bigen
        self.humor = humor
        self.loss_on_both = loss_on_both
        self.force_loss_on_jfeats = force_loss_on_jfeats
        self.ablation_no_kl_combine = ablation_no_kl_combine
        self.ablation_no_kl_gaussian = ablation_no_kl_gaussian
        self.ablation_no_motionencoder = ablation_no_motionencoder
        self.contrast= contrast
        self.fskate = fskate

        losses = []
        if mode == "xyz" or force_loss_on_jfeats:
            if not ablation_no_motionencoder:
                losses.append("recons_jfeats2jfeats")
            losses.append("recons_text2jfeats")
        if mode == "smpl":
            if not ablation_no_motionencoder:
                losses.append("recons_rfeats2rfeats")
            losses.append("recons_text2rfeats")
            losses.append("recons_connect2connect")
        else:
            ValueError("This mode is not recognized.")

        if vae or loss_on_both:
            kl_losses = []
            if not ablation_no_kl_combine and not ablation_no_motionencoder:
                kl_losses.extend(["kl_text2motion", "kl_motion2text"])
            if not ablation_no_kl_gaussian:
                if ablation_no_motionencoder:
                    kl_losses.extend(["kl_text"])
                else:
                    kl_losses.extend(["kl_text", "kl_motion","kl_connect"])
            losses.extend(kl_losses)
        if not self.vae or loss_on_both:
            if not ablation_no_motionencoder:
                losses.append("latent_manifold")
        if self.discriminator:
            losses.append("pred_clf")
            losses.append("gt_clf")
        if self.bigen:
            losses.append("recons_text2text")
            losses.append("recons_rfeats2text")
        if self.humor:
            losses.extend(["kl_text2humor","latent_humor","recons_text2humor"])
        
        if self.contrast:
            losses.extend(["text_contrast","motion_contrast"])
        
        if self.fskate:
            losses.append("footskate")
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

    def update(self, ds_text=None, ds_motion=None, ds_connect=None, ds_ref=None,
               ds_ref_connect = None, lat_text=None, lat_motion=None, lat_connect = None,
               dis_text=None, dis_motion=None, dis_connect=None, dis_ref=None, dis_ref_connect=None, 
               ref_probs= None, pred_probs=None, motion2text=None, text2text=None, textref=None,
               dis_humor = None, lat_humor = None, ds_humor = None, weights=None, lat_motion_contrast=None,
               lat_text_contrast= None, dis_text_contrast=None, dis_motion_contrast=None, lengths=None ):
        total: float = 0.0
        bs = ds_motion.rfeats.size(0)
        nframes = ds_motion.rfeats.size(1)
        feat_degree = ds_motion.rfeats.size(2)
        if self.mode == "xyz" or self.force_loss_on_jfeats:
            if not self.ablation_no_motionencoder:
                total += self._update_loss("recons_jfeats2jfeats", ds_motion.jfeats, ds_ref.jfeats)
            total += self._update_loss("recons_text2jfeats", ds_text.jfeats, ds_ref.jfeats)

        if self.mode == "smpl":
            if not self.ablation_no_motionencoder:
                if weights is None:
                    total += self._update_loss("recons_rfeats2rfeats", ds_motion.rfeats, ds_ref.rfeats)
                else:
                    stack_weights = weights.view(-1,1,1).repeat(1,nframes,feat_degree)
                    total += self._update_loss("recons_rfeats2rfeats", ds_motion.rfeats*stack_weights, ds_ref.rfeats*stack_weights)
            if weights is None:
                total += self._update_loss("recons_text2rfeats", ds_text.rfeats, ds_ref.rfeats)
                total += self._update_loss("recons_text2rfeats", ds_text.joints, ds_ref.joints)
                total += self._update_loss("recons_connect2connect",ds_connect.rfeats,ds_ref_connect.rfeats)
            else:
                stack_weights = weights.view(-1,1,1).repeat(1,nframes,feat_degree)
                total += self._update_loss("recons_text2rfeats", ds_text.rfeats*stack_weights, ds_ref.rfeats*stack_weights)
                stack_weights = weights.view(-1,1,1,1).repeat(1,nframes,21,3)
                total += self._update_loss("recons_text2rfeats", ds_text.joints*stack_weights, ds_ref.joints*stack_weights)
                stack_weights = weights.view(-1,1,1).repeat(1,ds_connect.rfeats.size(1),feat_degree)
                total += self._update_loss("recons_connect2connect",ds_connect.rfeats*stack_weights,ds_ref_connect.rfeats*stack_weights)
            
        if self.vae or self.loss_on_both:
            if not self.ablation_no_kl_combine and not self.ablation_no_motionencoder:
                total += self._update_loss("kl_text2motion", dis_text, dis_motion, weights)
                total += self._update_loss("kl_motion2text", dis_motion, dis_text, weights)
            if not self.ablation_no_kl_gaussian:
                total += self._update_loss("kl_text", dis_text, dis_ref, weights)
                if not self.ablation_no_motionencoder:
                    total += self._update_loss("kl_motion", dis_motion, dis_ref, weights)
                total += self._update_loss("kl_connect",dis_connect, dis_ref_connect, weights)
        
        if not self.vae or self.loss_on_both:
            if not self.ablation_no_motionencoder:
                if weights is None:
                    total += self._update_loss("latent_manifold", lat_text, lat_motion)
                else:
                    stack_weights = weights.view(-1,1).repeat(1,lat_text.size(-1))
                    total += self._update_loss("latent_manifold", lat_text*stack_weights, lat_motion*stack_weights)

        if self.discriminator:
            ref_labels = torch.ones_like(ref_probs)
            pred_labels = torch.zeros_like(pred_probs)
            total += self._update_loss("gt_clf", ref_probs, ref_labels)
            total += self._update_loss("pred_clf", pred_probs, pred_labels)
        
        if self.bigen:
            if weights is None:
                total += self._update_loss("recons_text2text", text2text, textref)
                total += self._update_loss("recons_rfeats2text", motion2text, textref)
            else:
                stack_weights = weights.view(-1,1,1).repeat(1,nframes,feat_degree)
                total += self._update_loss("recons_text2text", text2text*stack_weights, textref*stack_weights)
                total += self._update_loss("recons_rfeats2text", motion2text, textref)
        
        if self.humor:
            total += self._update_loss("kl_text2humor", dis_connect, dis_humor, weights)
            if weights is None:
                total += self._update_loss("recons_text2humor", ds_connect.rfeats, ds_humor.rfeats)
                total += self._update_loss("latent_humor", lat_connect, lat_humor)
            else:
                stack_weights = weights.view(-1,1,1).repeat(1,ds_connect.rfeats.size(1),feat_degree)
                total += self._update_loss("recons_text2humor", ds_connect.rfeats*stack_weights,ds_humor.rfeats*stack_weights)
                stack_weights = weights.view(-1,1).repeat(1,lat_connect.size(-1))
                total += self._update_loss("latent_humor", lat_connect*stack_weights, lat_humor*stack_weights)
        
        if self.contrast:
            total += self._update_loss("text_contrast", lat_text_contrast, lat_text)
            total += self._update_loss("motion_contrast", lat_motion_contrast, lat_motion)
        
        if self.fskate:
            total += self._update_loss("footskate", ds_text.joints, ds_ref.joints, lengths)
        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = self.count
        # return {loss: self.losses_values[loss]/count for loss in self.losses}
        return {loss: getattr(self, loss)/count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs, weights=None):
        # Update the loss
        if weights is None:
            val = self._losses_func[loss](outputs, inputs)
        else:
            val = self._losses_func[loss](outputs, inputs, weights)
        # self.losses_values[loss] += val.detach()
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        print(loss,weighted_loss)
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        elif loss == "footskate":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
