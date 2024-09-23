###
# Author: Kai Li
# Date: 2022-05-26 18:09:54
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:00:28
###
import gc
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from omegaconf import ListConfig

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        discriminator=None,
        optimizer=None,
        loss_func=None,
        metrics=None,
        scheduler=None,
    ):
        super().__init__()
        self.audio_model = model
        self.discriminator = discriminator
        self.optimizer = list(optimizer)
        self.loss_func = loss_func
        self.metrics = metrics
        self.scheduler = list(scheduler)
        
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss"
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False

    def forward(self, wav):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.audio_model(wav)

    def training_step(self, batch, batch_nb):
        ori_data, codec_data = batch
        optimizer_g, optimizer_d = self.optimizers()
        # multiple schedulers
        scheduler_g, scheduler_d = self.lr_schedulers()
        
        # train discriminator
        optimizer_g.zero_grad()
        output = self(codec_data)
        
        optimizer_d.zero_grad()
        est_outputs, _ = self.discriminator(output.detach(), sample_rate=44100)
        target_outputs, _ = self.discriminator(ori_data, sample_rate=44100)
        
        loss_d = self.loss_func["d"](target_outputs, est_outputs)
        self.manual_backward(loss_d)
        self.clip_gradients(optimizer_d, gradient_clip_val=5, gradient_clip_algorithm="norm")
        optimizer_d.step()
        # train generator
        est_outputs, est_feature_maps = self.discriminator(output, sample_rate=44100)
        _, targets_feature_maps = self.discriminator(ori_data, sample_rate=44100)
        
        loss_g = self.loss_func["g"](est_outputs, est_feature_maps, targets_feature_maps, output, ori_data)
        self.manual_backward(loss_g)
        self.clip_gradients(optimizer_g, gradient_clip_val=5, gradient_clip_algorithm="norm")
        optimizer_g.step()
        # print(loss)
        
        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_d.step()

        self.log(
            "train_loss_d",
            loss_d,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        
        self.log(
            "train_loss_g",
            loss_g,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )


    def validation_step(self, batch, batch_nb):
        # cal val loss
        ori_data, codec_data = batch
        # print(mixtures.shape)
        est_sources = self(codec_data)
        loss = self.metrics(est_sources, ori_data)
        
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        
        self.validation_step_outputs.append(loss)
        
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer[0].param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"val_pit_sisnr": -val_loss, "epoch": self.current_epoch}
        )

        self.validation_step_outputs.clear()  # free memory
        torch.cuda.empty_cache()
        
    def test_step(self, batch, batch_nb):
        mixtures, targets = batch
        est_sources = self(mixtures)
        loss = self.metrics(est_sources, targets)
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.test_step_outputs.append(loss)
        return {"test_loss": loss}
    
    def on_test_epoch_end(self):
        # val
        avg_loss = torch.stack(self.test_step_outputs).mean()
        test_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
        )

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer
        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers
            
        if not isinstance(self.optimizer, (list, tuple)):
            self.optimizer = [self.optimizer]  # support multiple schedulers
        
        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return self.optimizer, epoch_schedulers
            
    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
