import logging
import os
from typing import Optional

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
from .generator import BatchGenerator

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


class GaussianDiffusionBatchGenerator(BatchGenerator):
    """Imputator using gaussian diffusion to fill in missing data"""

    diffusion_batch: int
    """Number of images to sample from the diffusion model at once"""

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

    def __init__(self, diffusion_batch, config_file: str = "CoPaint/configs/celebahq.yaml",
                 device: Optional[torch.device] = None, **kwargs):
        self.diffusion_batch = diffusion_batch

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
        return f"CoPaint {self.conf['algorithm']} Imputator"

    @override
    def createBatch(self, image: Tensor, samples: int, index: int = None, rand: Generator = None) -> Tensor:
        """
        Creates a batch of images for the given passed image
        :param image:   Original image
        :param rand:    Random state
        :param index:   Index of the sample, for use in caching results. If none then no cache is possible
        :param samples: Number of samples to take. If unset, fetches from the class fields
        :return:  Batch of images based on samples
        """
        batch = image.repeat(samples, 1, 1, 1)
        diffusion_batches = samples // self.diffusion_batch
        last_diffusion_batch = samples % self.diffusion_batch

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
        for bIdx in range(diffusion_batches):
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
        if last_diffusion_batch != 0:
            model_kwargs = {
                "gt": image.repeat(last_diffusion_batch, 1, 1, 1),
                "gt_keep_mask": mask.repeat(last_diffusion_batch, 1, 1, 1),
            }
            result = self.sampler.p_sample_loop(
                self.model_fn,
                shape=(last_diffusion_batch, 3, self.image_size, self.image_size),
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=image.device,
                progress=True,
                return_all=True,
                conf=self.conf,
                sample_dir=None,
            )
            startIdx = diffusion_batches * self.diffusion_batch
            batch[startIdx:(startIdx + last_diffusion_batch), missing] = result["sample"][:, missing]
        return batch
