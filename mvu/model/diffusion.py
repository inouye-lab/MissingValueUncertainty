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
from ..serializer import loadValue, saveValue

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

    def __init__(self, regressor: Regressor, samples: int,
                 *args,  # want the rest to only be named arguments
                 diffusion_batch: int = None,
                 config_file: str = "CoPaint/configs/celebahq.yaml",
                 device: Optional[torch.device] = None,
                 **kwargs):
        self.regressor = regressor
        self.samples = samples
        self.diffusion_batch = diffusion_batch if diffusion_batch is not None else samples

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

    def createBatch(self, image: Tensor, rand: Generator = None, index: int = None, samples: int = None) -> Tensor:
        """
        Creates a batch of images for the given passed image
        :param image:   Original image
        :param rand:    Random state
        :param index:   Index of the sample, for use in caching results. If none then no cache is possible
        :param samples: Number of samples to take. If unset, fetches from the class fields
        :return:  Batch of images based on samples
        """
        if samples is None:
            samples = self.samples
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

    @override
    def predictWithUncertainty(self, features: Tensor, rand: Generator = None, index: int = None
                               ) -> Tuple[Tensor, Tensor]:
        featureSamples = features.shape[INDEX_SAMPLE]
        means: Optional[Tensor] = None
        variances: Optional[Tensor] = None

        for fIdx in range(features.shape[INDEX_SAMPLE]):
            batch = self.createBatch(features[fIdx], rand, index)
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


class CachedGaussianDiffusionMethod(GaussianDiffusionMethod):
    cachePath: str
    """Path to the folder containing the cached batches"""
    cacheMask: Tensor
    """Mask used for the cache, important it matches for the index to be valid"""

    def __init__(self, regressor: Regressor, samples: int, cache_path: str, cache_mask: Tensor,
                 device: Optional[torch.device] = None, **kwargs):
        super().__init__(regressor, samples, device=device, **kwargs)
        self.cachePath = cache_path
        self.cacheMask = cache_mask.to(device)
        # ensure the sample directory exists
        os.makedirs(self.cachePath, exist_ok=True)
        # ensure the mask in the sample directory matches the mask passed, if not this directory will cause issues
        maskPath = os.path.join(self.cachePath, "mask.pklz")
        if os.path.exists(maskPath):
            directoryMask = loadValue(maskPath, Tensor)
            assert torch.equal(cache_mask.cpu(), directoryMask), \
                f"Directory mask mismatches: passed {self.cacheMask}, but directory contains {directoryMask}"
        else:
            # if the directory lacks a mask, it is probably new, so just save our mask
            saveValue(cache_mask.cpu(), maskPath, Tensor)

    @override
    def createBatch(self, image: Tensor, rand: Generator = None, index: int = None, samples: int = None) -> Tensor:
        if samples is None:
            samples = self.samples
        if index is None:
            logging.info("No sample index passed, skipping cache.")
        elif not torch.equal(self.cacheMask, torch.isnan(image)):
            logging.error(f"Image {index} mask does not match the method's mask, unable to use cache")
        else:
            # caching is possible, do we have a cached value?
            batchPath = os.path.join(self.cachePath, f"{index}.pklz")
            if not os.path.exists(batchPath):
                logging.info(f"No cache found for sample {index}, generating new batch.")
            else:
                # cache is valid, use that
                cached = loadValue(batchPath, Tensor).to(image.device)
                if cached.shape[0] >= samples:
                    # note the cache may contain more samples than requested, that is fine, just take the number requested
                    # means it was created with a different variant of this method
                    # TODO: consider random indexing instead?
                    return cached[0:samples]
                neededSamples = samples - cached.shape[0]
                logging.info(f"Image {index} only has {cached.shape[0]} cached samples, computing {neededSamples} additional samples")
                batch = super().createBatch(image, rand, index, samples - cached.shape[0])
                # combine the two sets and cache the larger number of samples
                batch = torch.cat((cached, batch), dim=0)
                saveValue(batch.cpu(), batchPath, Tensor)
                # no need to index, size is calculated exactly
                return batch
        batch = super().createBatch(image, rand, index, samples)
        saveValue(batch.cpu(), batchPath, Tensor)
        return batch
