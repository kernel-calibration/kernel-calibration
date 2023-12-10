from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.calibration_error import CalibrationError
from src.metrics import ShannonEntropyError, ClassificationKernelCalibrationError

from typing import Any, List, Literal, Optional, Dict, Callable

class ClassificationLitModule(LightningModule):
    """ LightningModule for Classification tasks.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: Callable,
        calibrator: Callable = None,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        kcal_kwargs = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = criterion
        self.calibrator = calibrator

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # Initialize metrics
        assert net.output_size >= 2, f"Must have >=2 classes for classification task. Model only has {net.output_size} classes."
        task = 'binary' if net.output_size == 2 else 'multiclass'
        ece_kwargs = {"task": task, "n_bins": 20, "norm": 'l1', "num_classes": net.output_size}
        self.train_ece = CalibrationError(**ece_kwargs)
        self.val_ece = CalibrationError(**ece_kwargs)
        self.test_ece = CalibrationError(**ece_kwargs)

        self.train_entropy = ShannonEntropyError()
        self.val_entropy = ShannonEntropyError()
        self.test_entropy = ShannonEntropyError()

        if kcal_kwargs is None:
            kcal_kwargs = {}
        self.test_kcal = ClassificationKernelCalibrationError(**kcal_kwargs)

        # For logging best validation metrics
        self.val_acc_best = MaxMetric()
        self.val_ece_best = MinMetric()
        self.val_entropy_best = MinMetric()

        # Additional metrics for post-hoc calibration
        if self.calibrator:
            self.test_calibrated_acc = Accuracy()
            self.test_calibrated_ece = CalibrationError(**ece_kwargs)
            self.test_calibrated_entropy = ShannonEntropyError()
            self.test_calibrated_kcal = ClassificationKernelCalibrationError(**kcal_kwargs)


    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_ece_best.reset()

    def step(self, batch: Any):
        x, y = batch
        y = y.squeeze(-1)
        logits = self.forward(x)

        loss = self.criterion(x, y ,logits)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, logits, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, logits, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        ece = self.train_ece(logits, targets)
        entropy = self.train_entropy(logits)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/entropy", entropy, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, logits, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        ece = self.val_ece(logits, targets)
        entropy = self.val_entropy(logits)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/entropy", entropy, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best(acc)

        ece = self.val_ece.compute()  # get val accuracy from current epoch
        self.val_ece_best(ece)

        entropy = self.val_entropy.compute()  # get val accuracy from current epoch
        self.val_entropy_best(entropy)

        # log `*_best` metrics as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/ece_best", self.val_ece_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/entropy_best", self.val_entropy_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_start(self):
        if self.calibrator is None:
            return

        val_x, val_y = self.trainer.datamodule.data_val[:]
        val_x, val_y = val_x.to(self.device), val_y.squeeze(-1).to(self.device)

        with torch.no_grad():
            logits = self.forward(val_x)
            val_pred = F.softmax(logits, dim=-1)

        with torch.enable_grad():
            self.calibrator.train(val_pred, val_y)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, logits, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        ece = self.test_ece(logits, targets)
        entropy = self.test_entropy(logits)
        
        x, _ = batch
        kcal = self.test_kcal(logits, targets, x)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/entropy", entropy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/kcal", kcal, on_step=False, on_epoch=True, prog_bar=True)

        # If post-hoc calibration method is chosen, apply it to model predictions
        if self.calibrator:
            pred_dist = F.softmax(logits, dim=-1)
            with torch.no_grad():
                calibrated_dists = self.calibrator(pred_dist)
                calibrated_preds = torch.argmax(calibrated_dists, dim=-1)
            
            calibrated_acc = self.test_calibrated_acc(calibrated_preds, targets)
            calibrated_ece = self.test_calibrated_ece(calibrated_dists, targets)
            calibrated_entropy = self.test_calibrated_entropy(calibrated_dists, is_dist=True)
            calibrated_kcal = self.test_kcal(calibrated_dists, targets, x)

            # log post-hoc calibrated test metrics
            self.log(f"test/calibrated_acc", calibrated_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_ece ", calibrated_ece , on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_entropy", calibrated_entropy, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_kcal", calibrated_kcal, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # Reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

        self.train_ece.reset()
        self.test_ece.reset()
        self.val_ece.reset()

        self.train_entropy.reset()
        self.test_entropy.reset()
        self.val_entropy.reset()

        self.test_kcal.reset()

        if self.calibrator:
            self.test_calibrated_acc.reset()
            self.test_calibrated_ece.reset()
            self.test_calibrated_entropy.reset()
            self.test_calibrated_kcal.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
