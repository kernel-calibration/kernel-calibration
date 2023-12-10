from typing import Any, List, Literal, Optional, Dict, Callable

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from src.metrics import GMMCalibrationError, GMMNegativeLogLikelihood, GMMKernelCalibrationError, DecisionCalibrationError
from src.utils import gmm_nll_dummy_x, gmm_params_to_dist
from src.models.components.simple_dense_net import SimpleDenseNet
from torchuq.transform.conformal import ConformalCalibrator

class GMMLitModule(LightningModule):
    """ LightningModule for Regression tasks. Uses a Gaussian Mixture Model (GMM) forecaster.

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
    ):
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = criterion
        self.calibrator = calibrator

        # Use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_nll = GMMNegativeLogLikelihood()
        self.val_nll = GMMNegativeLogLikelihood()
        self.test_nll = GMMNegativeLogLikelihood()

        self.train_cal = GMMCalibrationError()
        self.val_cal = GMMCalibrationError()
        self.test_cal = GMMCalibrationError()

        kcal_kwargs = {"operands": {'x': "rbf", 'y': "rbf"}, "scalers": {'p': 1, 'x': 1, 'y': 1}, "bandwidth": 10, "num_samples": 10}
        self.train_kcal = GMMKernelCalibrationError(**kcal_kwargs)
        self.val_kcal = GMMKernelCalibrationError(**kcal_kwargs)
        self.test_kcal = GMMKernelCalibrationError(**kcal_kwargs)

        dcal_kwargs = {"loss_fn_cls": "loss_fn_1", "actions": [-1, 1], "num_samples": 10, "metric": "L2"}
        self.train_dcal = DecisionCalibrationError(**dcal_kwargs)
        self.val_dcal = DecisionCalibrationError(**dcal_kwargs)
        self.test_dcal = DecisionCalibrationError(**dcal_kwargs)

        # For logging best so far validation accuracy
        self.val_nll_best = MinMetric()
        self.val_cal_best = MinMetric()
        self.val_kcal_best = MinMetric()
        self.val_dcal_best = MinMetric()

        # Additional metrics for post-hoc calibration
        if self.calibrator:
            self.test_calibrated_nll = GMMNegativeLogLikelihood(use_dist=True)
            self.test_calibrated_cal = GMMCalibrationError(use_dist=True)
            self.test_calibrated_kcal = GMMKernelCalibrationError(**kcal_kwargs, use_dist=True)
            self.test_calibrated_dcal = DecisionCalibrationError(**dcal_kwargs, use_dist=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_nll_best.reset()
        self.val_cal_best.reset()
        self.val_kcal_best.reset()
        self.val_dcal_best.reset()

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(x, y, preds)
        return loss, preds, x, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, inputs, targets = self.step(batch)

        # log train metrics
        nll = self.train_nll(preds, targets)
        cal = self.train_cal(preds, targets)
        kcal = self.train_kcal(preds, targets, inputs)
        dcal = self.train_dcal(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/nll", nll, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/cal", cal, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/kcal", kcal, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/dcal", dcal, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, inputs, targets = self.step(batch)

        # log val metrics
        nll = self.val_nll(preds, targets)
        cal = self.val_cal(preds, targets)
        kcal = self.val_kcal(preds, targets, inputs)
        dcal = self.val_dcal(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/nll", nll, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cal", cal, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/kcal", kcal, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/dcal", dcal, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        nll = self.val_nll.compute()  # get val accuracy from current epoch
        self.val_nll_best.update(nll)
        self.log("val/nll_best", self.val_nll_best.compute(), on_epoch=True, prog_bar=True, sync_dist=True)

        cal = self.val_cal.compute()  # get cal accuracy from current epoch
        self.val_cal_best.update(cal)
        self.log("val/cal_best", self.val_cal_best.compute(), on_epoch=True, prog_bar=True, sync_dist=True)

        kcal = self.val_kcal.compute()  # get kcal accuracy from current epoch
        self.val_kcal_best.update(kcal)
        self.log("val/kcal_best", self.val_kcal_best.compute(), on_epoch=True, prog_bar=True, sync_dist=True)

        dcal = self.val_dcal.compute()  # get dcal accuracy from current epoch
        self.val_dcal_best.update(dcal)
        self.log("val/dcal_best", self.val_dcal_best.compute(), on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        if self.calibrator is None:
            return

        val_x, val_y = self.trainer.datamodule.data_val[:]
        val_x, val_y = val_x.to(self.device), val_y.to(self.device)

        with torch.no_grad():
            pred_raw = self.net(val_x)

        mean, std, prob = pred_raw
        val_pred = gmm_params_to_dist(mean.squeeze(1), std.squeeze(1), prob)
        self.calibrator.train(val_pred, val_y)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, inputs, targets = self.step(batch)

        nll = self.test_nll(preds, targets)
        cal = self.test_cal(preds, targets)
        kcal = self.test_kcal(preds, targets, inputs)
        dcal = self.test_dcal(preds, targets)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/nll", nll, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/cal", cal, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/kcal", kcal, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/dcal", dcal, on_step=False, on_epoch=True, sync_dist=True)

        # if post-hoc calibration method is chosen, apply it to preds
        if self.calibrator:
            mean, std, prob = preds
            dist_preds = gmm_params_to_dist(mean.squeeze(1), std.squeeze(1), prob)

            with torch.no_grad():
                preds = self.calibrator(dist_preds)
            
            calibrated_nll = self.test_calibrated_nll(preds, targets.squeeze(1))
            calibrated_cal = self.test_calibrated_cal(preds, targets.squeeze(1))
            calibrated_kcal = self.test_calibrated_kcal(preds, targets.squeeze(1), inputs)
            calibrated_dcal = self.test_calibrated_dcal(preds, targets.squeeze(1))

            # log post-hoc calibrated test metrics
            self.log(f"test/calibrated_nll", calibrated_nll, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/calibrated_cal", calibrated_cal, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/calibrated_kcal", calibrated_kcal, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/calibrated_dcal", calibrated_dcal, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        nll = self.test_nll.compute()  # get val accuracy
        cal = self.test_cal.compute()  # get cal accuracy
        kcal = self.test_kcal.compute()  # get kcal accuracy
        dcal = self.test_dcal.compute() # get decision loss

        self.log(f"test/nll", nll, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/cal", cal, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/kcal", kcal, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/dcal", dcal, on_step=False, on_epoch=True, sync_dist=True)

        if self.calibrator:
            calibrated_nll = self.test_calibrated_nll.compute() # get post-hoc calibrated val accuracy
            calibrated_cal = self.test_calibrated_cal.compute() # get post-hoc calibrated cal accuracy
            calibrated_kcal = self.test_calibrated_kcal.compute() # get post-hoc calibrated kcal accuracy
            calibrated_dcal = self.test_calibrated_dcal.compute() # get post-hoc calibrated decision loss

            self.log(f"test/calibrated_nll", calibrated_nll, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/calibrated_cal", calibrated_cal, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/calibrated_kcal", calibrated_kcal, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/calibrated_dcal", calibrated_dcal, on_step=False, on_epoch=True, sync_dist=True)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_nll.reset()
        self.test_nll.reset()
        self.val_nll.reset()

        self.train_cal.reset()
        self.test_cal.reset()
        self.val_cal.reset()

        self.train_kcal.reset()
        self.test_kcal.reset()
        self.val_kcal.reset()

        self.train_dcal.reset()
        self.test_dcal.reset()
        self.val_dcal.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
