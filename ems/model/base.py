import numpy as np
from pytorch_lightning import LightningModule


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

        # Need to define:
        # forward
        # allsplit_step()
        # metrics()
        # losses()
        # optimizer

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        output_loss = self.allsplit_step("val", batch, batch_idx)
        # metrics_dict = self.metrics.compute()
        # dico = {}
        # dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})

        # self.log_dict(dico,batch_size = len(batch["length"]),sync_dist=True)
        return output_loss


    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def allsplit_epoch_end(self, split: str, outputs):
        losses = self.losses[split]
        loss_dict = losses.compute(split)
        dico = {losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items()}
        if split == "val":
            metrics_dict = self.metrics.compute()
            dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})
        dico.update({"epoch": float(self.trainer.current_epoch),
                     "step": float(self.trainer.current_epoch)})
        self.log_dict(dico,sync_dist=True)

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        return self.allsplit_epoch_end("test", outputs)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}
