import collections.abc
from pathlib import Path

import more_itertools as mitertools
import numpy as np
import torch
from natsort import natsorted, ns
from numpy import linalg as LA
from torch.utils.data import DataLoader, Dataset

from quantaodometry.color import (binary2rgb, linearrgb_to_srgb,
                                  srgb_to_linearrgb)
from quantaodometry.homography import get_max_extent, sample_warped_frames
from quantaodometry.io import read_img
from quantaodometry.utils import img_to_tensor, to_tensor, translation_matrix


class Patches:
    """Generate warped patches from source image `img` of size `crop_size` using homographies `hs`."""

    def __init__(self, img, hs, crop_size, transform=None, indices=None, step=1, device=None, **kwargs):
        self.extra_offset = torch.zeros(2)

        if indices is not None and step != 1:
            raise ValueError("Only one of `step`, `indices` can be specified!")
        self.indices = indices or range(0, len(hs), step)
        self.img, self.hs, self.device, self.crop_size = img, hs, device, crop_size
        self.shifted_hs = [None] * len(hs)
        self.slices = [None] * len(hs)
        self.transform = transform

    @classmethod
    def from_directory(cls, root_dir, pattern="*.png", crop_size=None, **kwargs):
        paths = natsorted(Path(root_dir).rglob(pattern), alg=ns.PATH)

        if not paths:
            raise RuntimeError(f"No files matching `{pattern}` where found in {root_dir} or sub-directories.")

        return cls(None, paths, crop_size, **kwargs)

    def as_dataset(self):
        return PatchDataset(self)

    def warp_single(self, h, img):
        img = img_to_tensor(img, batched=True, device=self.device)
        return sample_warped_frames([img], h, self.crop_size, device=self.device)[0]

    def get_base_patch(self, item):
        """Return the smallest patch from which the desired patch can be sampled from as well as the shifted
        warp that accomplishes this sampling. For more see `warp_smallest` in the `Panorama` class."""
        if self.img is None:
            # Re-use hs as image paths
            img, _ = read_img(self.hs[item])
            return item, self.hs[item], img_to_tensor(img, device=self.device)

        if self.slices[item] is None:
            # Get the extent of the warp, which likely is non-integer, and round it up so it's whole.
            # The extra shift due to the fractional part is accounted for while warping below.
            device = getattr(self.img, "device", None)
            hs = translation_matrix(-self.extra_offset.numpy()) @ self.hs[item]
            (height, width), offset = get_max_extent(
                LA.inv(hs).reshape(1, 3, 3), [self.crop_size], ceil=False, device=device
            )
            padding = padding_x, padding_y = torch.frac(offset)
            height, width = torch.ceil(height + padding_y).to(int).item(), torch.ceil(width + padding_x).to(int).item()
            crop_offset = (offset - padding).to(int)

            start_w, start_h = crop_offset.tolist()
            self.slices[item] = np.s_[start_h : start_h + height, start_w : start_w + width]
            self.shifted_hs[item] = translation_matrix(-crop_offset) @ to_tensor(hs, device=device)

        if torch.is_tensor(self.img):
            img = self.img[(..., *self.slices[item])]
        else:
            np_img = self.img[self.slices[item]]
            img = img_to_tensor(np_img / 255).squeeze()
        return item, self.shifted_hs[item], img

    def finalize_patch(self, idx, h, base_patch):
        """Apply any final transformations to base patch (warp, to-device, sampling, etc)"""
        if self.img is None:
            return base_patch
        return self.warp_single(h, base_patch)

    def post_merge_transform(self, patch, *args, **kwargs):
        """Transformation to apply to merged patches before using them to estimate warps"""
        return patch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        item = self.indices[item]
        frame = self.finalize_patch(*self.get_base_patch(item))

        if self.transform is not None:
            frame = self.transform(frame)

        return frame


class BinaryPatches(Patches):
    """Deterministically generate binary patches from source image `img` of size `crop_size`
    using homographies `hs`. Importantly, this allows for random accesses to any patch without generating
    all patches preceding it.
    """

    def __init__(
        self, img, hs, crop_size, factor=1, device=None, is_linear=False, tonemap_merge=True, seed=2147483647, **kwargs
    ):
        super().__init__(img, hs, crop_size, device=device, **kwargs)
        generator = torch.Generator().manual_seed(seed)
        self.seeds = torch.randint(0, int(1e10), size=(len(hs),), generator=generator)
        self.tonemap_merge = tonemap_merge
        self.is_linear = is_linear
        self.factor = factor

    def sample_single(self, warped_img, factor=1.0, seed=12345):
        rng = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.bernoulli(1 - torch.exp(-warped_img * factor), generator=rng)

    def finalize_patch(self, idx, h, base_patch):
        """Apply any final transformations to base patch (warp, to-device, sampling, etc)"""
        warped_img = super().finalize_patch(idx, h, base_patch)

        # Ensure intensities are linearized before bernoulli sampling
        if not self.is_linear:
            warped_img = srgb_to_linearrgb(warped_img)

        return self.sample_single(warped_img, factor=self.factor, seed=self.seeds[idx])

    def post_merge_transform(self, patch, *args, **kwargs):
        patch = binary2rgb(patch, factor=self.factor)
        if self.tonemap_merge:
            patch = linearrgb_to_srgb(patch)
        return patch


class RGBPatches(Patches):
    """Same as `Patches` except that the `post_merge_transform` simulates a real camera."""

    def __init__(
        self,
        img,
        hs,
        crop_size,
        fwc=500,
        readout_std=20,
        factor=1,
        device=None,
        is_linear=False,
        seed=2147483647,
        **kwargs,
    ):
        super().__init__(img, hs, crop_size, device=device, **kwargs)
        self.generator = torch.Generator().manual_seed(seed)
        self.readout_std = readout_std
        self.is_linear = is_linear
        self.factor = factor
        self.fwc = fwc

    def finalize_patch(self, idx, h, base_patch):
        """Apply any final transformations to base patch (warp, to-device, sampling, etc)"""
        warped_img = super().finalize_patch(idx, h, base_patch)

        if self.is_linear:
            return warped_img * self.factor
        return srgb_to_linearrgb(warped_img) * self.factor

    def post_merge_transform(self, patch, *args, burst_size=None, **kwargs):
        seed = int(torch.randint(1, (1,), generator=self.generator))
        return self.emulate_from_merged(
            patch,
            burst_size=burst_size,
            readout_std=self.readout_std,
            fwc=self.fwc,
            factor=self.factor,
            generator=torch.Generator(device=patch.device).manual_seed(seed),
        )

    @staticmethod
    def emulate_from_merged(patch, burst_size=200, readout_std=20, fwc=500, factor=1.0, generator=None):
        # Input patch is average of `burst_size` linear-intensity frames, get sum by multiplying.
        patch = patch * burst_size

        # Above sum is in range [0, burst_size*factor]
        # Perform poisson sampling and add zero-mean gaussian read noise
        patch = torch.poisson(patch, generator=generator)
        patch += torch.normal(torch.zeros_like(patch), readout_std, generator=generator)

        # Normalize by full well capacity, clip highlights, and quantize to 12-bits
        patch = torch.clip(patch / fwc, 0, 1.0)
        patch = torch.round(patch * 2**12) / 2**12

        # Multiply by gain to keep constant(-ish) brightness
        patch *= fwc / (burst_size * factor)

        # Convert to sRGB color space for viewing and quantize to 8-bits
        patch = linearrgb_to_srgb(patch)
        patch = torch.round(patch * 2**8) / 2**8
        return patch


class PatchDataset(Dataset):
    def __init__(self, patch_gen):
        self.patch_gen = patch_gen
        super().__init__()

    def as_dataset(self):
        return self

    def post_merge_transform(self, patch, *args, **kwargs):
        if hasattr(self.patch_gen, "post_merge_transform"):
            return self.patch_gen.post_merge_transform(patch, *args, **kwargs)
        return patch

    def __len__(self):
        return len(self.patch_gen)

    def __getitem__(self, item):
        return self.patch_gen[item]

    def __getattr__(self, item):
        return getattr(self.patch_gen, item)


class BatchedDataLoader:
    """Similar to torch.utils.data.DataLoader but uses multiple workers per batch.

    Pytorch's DataLoader uses one thread per batch and will prefetch batches
    if multiple workers/threads are used. This class breaks a batch into mini-batches,
    each of which is fetched by a single worker, and then merges these back together.

    IMPORTANT: The `collate_batch_fn` takes sub-batches and groups them together. A simple example of
        this is when each sub-batch is a tensor of shape (N,C,H,W) you might want to concatenate them
        together to (N',C,H,W). But, the argument `sub_batches` can be a generator. This happens when
        every thread is not yet ready, i.e: when workers_per_batch >> num_workers.
        In this scenario, if `collate_batch_fn` also return as iterator then we can avoid loading the
        whole batch into memory.

        TL;DR: To avoid loading the whole batch into memory (and instead only load `num_workers` amount
            of sub-batches), set `workers_per_batch` to a large number.

    Currently only supports batch_sampler mode.
    """

    def __init__(self, dataset, workers_per_batch=1, *args, batch_sampler=None, collate_batch_fn=None, **kwargs):
        if batch_sampler is None:
            raise ValueError("Required argument `batch_sampler` was not provided.")

        # It's important that batch_sampler is a Sequence, that is, it defines `__getitem__`/[].
        # If it isn't then calling next(iter(self)) will advance the loader by one step and any
        # subsequent calls won't start from the first item. In fact if there's more than one worker
        # it won't even start at item 1! Making `batch_sampler` a list is an easy, although
        # sub-optimal way to accomplish this...
        batch_sampler = list(
            mitertools.flatten((mitertools.divide(workers_per_batch, batch) for batch in batch_sampler))
        )
        self.collate_batch_fn = self.default_collate if collate_batch_fn is None else collate_batch_fn
        self.loader = DataLoader(dataset, *args, batch_sampler=batch_sampler, **kwargs)
        self.workers_per_batch = workers_per_batch
        self.dataset = dataset

    @staticmethod
    def default_collate(sub_batches):
        """Collates the resulting sub-batches from each worker into a full batch.
        Expects each sub-batch to be a tensor or sequence of tensors and simply concatenates them.
        """
        is_sequence = set(isinstance(sub_batch, collections.abc.Sequence) for sub_batch in sub_batches)

        if len(is_sequence) != 1:
            raise RuntimeError("Expected all sub-batches to be of the same type.")

        is_sequence = is_sequence.pop()
        sub_batches = [[sub_batch] if not is_sequence else sub_batch for sub_batch in sub_batches]

        lengths = set(len(sub_batch) for sub_batch in sub_batches)

        if len(lengths) != 1:
            raise ValueError(
                f"Expected all sub-batches to have the same number of elements " f"but found lengths {lengths}."
            )

        batch = [torch.concat(elements, dim=0) for elements in zip(*sub_batches)]
        return batch if is_sequence else batch[0]

    def __iter__(self):
        for batch in mitertools.ichunked(iter(self.loader), self.workers_per_batch):
            yield self.collate_batch_fn(batch)
