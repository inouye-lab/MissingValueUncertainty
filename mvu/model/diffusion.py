import logging
import os
from typing import Tuple, Optional

import torch
from overrides import override
from torch import Tensor, Generator

from CoPaint.guided_diffusion import O_DDIMSampler, DDIMSampler, DDNMSampler, DDRMSampler, DPSSampler, dist_util
from CoPaint.guided_diffusion.ddim import R_DDIMSampler
from CoPaint.guided_diffusion.respace import SpacedDiffusion
from CoPaint.guided_diffusion.script_util import diffusion_defaults, select_args, create_gaussian_diffusion, \
    create_model, model_defaults
from CoPaint.guided_diffusion.unet import UNetModel
from CoPaint.utils.config import Config
from .method import Method
from .regressor import Regressor
from ..dataset.meta import INDEX_SAMPLE

SAMPLER_CLS = {
    "repaint": SpacedDiffusion,
    "ddim": DDIMSampler,
    "o_ddim": O_DDIMSampler,
    "resample": R_DDIMSampler,
    "ddnm": DDNMSampler,
    "ddrm": DDRMSampler,
    "dps": DPSSampler,
}
"""
Sampler class options defined by CoPaint
"""


class GaussianDiffusionMethod(Method):
    """Imputator using gaussian diffusion to fill in missing data"""

    regressor: Regressor
    """Regressor to run on the results of this imputator"""
    samples: int
    """Number of Monte Carlo samples to take"""
    diffusion_batch: int
    """Number of images to sample from the diffusion model at once"""
    diffusion_batches: int
    """Number of batches needed to get all samples from diffusion"""
    last_diffusion_batch: int
    """Size of the final batch for diffusion, in case samples is not divisible by diffusion_batch"""

    conf: Config
    """CoPaint config instance"""
    image_size: int
    """Image size cached from the CoPaint config, as its used commonly"""
    sampler: SpacedDiffusion
    """Sampler algorithm"""
    unet: UNetModel
    """UNet instance used for the model function"""
    model_fn: callable
    """Model function for passing into the sampler"""

    def __init__(self, regressor: Regressor, samples: int, diffusion_batch: int = None,
                 config_file: str = "CoPaint/configs/celebahq.yaml",
                 device: Optional[torch.device] = None,
                 **kwargs):
        self.regressor = regressor
        self.samples = samples
        self.diffusion_batch = diffusion_batch if diffusion_batch is not None else samples
        self.diffusion_batches = self.samples // self.diffusion_batch
        self.last_diffusion_batch = self.samples % self.diffusion_batch

        self.conf = Config(default_config_file=config_file, default_config_dict=kwargs, use_argparse=False)
        self.image_size = self.conf["image_size"]
        self.sampler = create_gaussian_diffusion(
            **select_args(self.conf, diffusion_defaults().keys()),
            conf=self.conf,
            base_cls=SAMPLER_CLS[self.conf["algorithm"]],
        )

        self.unet = create_model(**select_args(self.conf, model_defaults().keys()), conf=self.conf)
        logging.info(f"Loading model from {self.conf['model_path']}...")
        self.unet.load_state_dict(
            dist_util.load_state_dict(
                os.path.expanduser(self.conf["model_path"]), map_location="cpu"
            ), strict=False
        )
        if device is not None:
            self.unet.to(device)
        if self.conf["use_fp16"]:
            self.unet.convert_to_fp16()
        self.unet.eval()
        if self.conf["class_cond"]:
            def model_fn(x, t, y=None, gt=None, **kwargs):
                return self.unet(x, t, y, gt=gt)
        else:
            def model_fn(x, t, gt=None, **kwargs):
                return self.unet(x, t, None, gt=gt)
        self.model_fn = model_fn

    @property
    @override
    def name(self) -> str:
        return "CoPaint Imputator"

    def createBatch(self, image: Tensor, rand: Generator = None) -> Tensor:
        """
        Creates a batch of images for the given passed image
        :param image:   Original image
        :param rand:    Random state
        :return:  Batch of images based on self.samples
        """
        batch = image.repeat(self.samples, 1, 1, 1)

        # sample diffusion model
        missing = torch.isnan(image)
        image = image.clone()
        # CoPaint does not support NaNs in the base image, so just zero it all out
        image[missing] = 0
        mask = (1 - missing.any(dim=0).float())
        model_kwargs = {
            "gt": image.repeat(self.diffusion_batch, 1, 1, 1),
            "gt_keep_mask": mask.repeat(self.diffusion_batch, 1, 1, 1),
        }
        print("Repeated ", model_kwargs["gt"].shape, model_kwargs["gt_keep_mask"].shape)
        for bIdx in range(self.diffusion_batches):
            # TODO: how do I use the generator here for seeding?
            result = self.sampler.p_sample_loop(
                self.model_fn,
                shape=(self.diffusion_batch, 3, self.image_size, self.image_size),
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=image.device,
                progress=True,
                return_all=True,
                conf=self.conf,
                sample_dir=None,
            )
            startIdx = bIdx * self.diffusion_batch
            batch[startIdx:(startIdx + self.diffusion_batch), missing] = result["sample"][:, missing]
        # its possible our batches don't divide evenly, so just sample the last one on its own
        if self.last_diffusion_batch != 0:
            model_kwargs = {
                "gt": image.repeat(self.last_diffusion_batch, 1, 1, 1),
                "gt_keep_mask": mask.repeat(self.last_diffusion_batch, 1, 1, 1),
            }
            result = self.sampler.p_sample_loop(
                self.model_fn,
                shape=(self.last_diffusion_batch, 3, self.image_size, self.image_size),
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=image.device,
                progress=True,
                return_all=True,
                conf=self.conf,
                sample_dir=None,
            )
            startIdx = self.diffusion_batches * self.diffusion_batch
            batch[startIdx:(startIdx + self.last_diffusion_batch), missing] = result["sample"][:, missing]
        return batch

    @override
    def predictWithUncertainty(self, features: Tensor, rand: Generator = None) -> Tuple[Tensor, Tensor]:
        featureSamples = features.shape[INDEX_SAMPLE]
        means: Optional[Tensor] = None
        variances: Optional[Tensor] = None

        for fIdx in range(features.shape[INDEX_SAMPLE]):
            batch = self.createBatch(features[fIdx], rand)
            prediction = self.regressor.predict(batch)
            # we don't know the output size without running the regressor, so lazily init the output tensors
            if means is None:
                means = torch.empty((featureSamples, *prediction.shape[1:]), device=features.device)
            if variances is None:
                variances = torch.empty((featureSamples, *prediction.shape[1:]), device=features.device)
            # fill in output from the prediction
            means[fIdx, :] = prediction.mean(dim=0)
            variances[fIdx, :] = prediction.var(dim=0)

        return means, variances
