import collections
import functools
import itertools
import math
import os
import warnings
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import more_itertools as mitertools
import numpy as np
import torch
from tqdm.auto import tqdm

from quantaodometry.localizers import compute_homographies

from .dataset import BatchedDataLoader
from .homography import (get_corners, get_max_extent, homography_interp,
                         identity_distance_transform, sample_warped_frames,
                         warp_points)
from .utils import (img_to_tensor, scale_matrix, tensor_to_img, to_cpu,
                    to_numpy, to_tensor, translation_matrix)


class DynamicCanvas(torch.nn.Module):
    """Infinitely extendable canvas.

    Much like an array, which has amortized insert times of O(1) yet doubles in size and gets
    all it's elements copied over every so often, this canvas can grow in any 2D direction as
    needed by allocating a new buffer and copying over old data using bilinear interpolation
    to allow for non-integer shifts.

    Assumes that at least one frame will be added with an identity warp. That is, if all frames
    that are added start at coordinates (1000,1000) the canvas will contain an empty region from
    the origin to (1000, 1000) which is sub-optimal.
    """

    def __init__(self, shape=(480, 640, 3), expand_coef=1.5, min_padding=1.0, max_pixels=200_000_000, **kwargs):
        super().__init__()
        self.height, self.width, *channels = shape
        self.channels = (channels[0] if channels else 1) + 1
        self.shape = (self.height, self.width, self.channels)
        self.max_pixels = max_pixels

        if expand_coef < 1.0:
            raise ValueError("Expansion coefficient must be greater than one!")

        coef = 2 * expand_coef - 1
        real_h = int(self.height * coef)
        real_w = int(self.width * coef)
        self.origin_init = torch.tensor(
            [(expand_coef - 1) * self.width, (expand_coef - 1) * self.height], dtype=torch.float32
        )
        self.register_buffer("data", torch.zeros((1, self.channels, real_h, real_w)))
        self.register_buffer("origin", self.origin_init.clone())
        self.expand_coef = expand_coef
        self.min_padding = min_padding
        self.resize_count = 0

    @classmethod
    def from_frame(cls, frame, **kwargs):
        obj = cls(frame.shape, **kwargs)
        retval = obj.add_frame(torch.eye(3), frame)
        return obj, *retval

    @classmethod
    def from_frames(cls, homographies, frames, shapes=None, pbar=True, online=False, **kwargs):
        if isinstance(frames, collections.abc.Iterator):
            (first_frame,), frames = mitertools.spy(frames)
        else:
            first_frame = frames[0]
        obj = cls(first_frame.shape, **kwargs)
        retval = obj.add_frames(homographies, frames, shapes=shapes, pbar=pbar, online=online)
        return obj, *retval

    @property
    def real_shape(self):
        """Shape of underlying allocated buffer"""
        *_, c, h, w = self.data.shape
        return h, w, c

    @property
    def footprint(self):
        """Return estimated memory footprint in bytes"""
        return self.data.element_size() * self.data.nelement()

    @property
    def total_shift(self):
        """Return total canvas shift from it's initialization"""
        return to_cpu(self.origin) - to_cpu(self.origin_init)

    def as_tensor(self):
        h, w, c = shape = np.ceil(self.shape).astype(int)
        (data,) = sample_warped_frames([self.data], translation_matrix(self.origin), shape, device=self.origin.device)
        return data[..., :h, :w]

    def as_tensor_full(self):
        return self.data

    def as_image(self):
        h, w, c = shape = np.ceil(self.shape).astype(int)
        data = self.as_tensor()
        img = tensor_to_img(to_cpu((data * 255).squeeze()))
        img = np.clip(img[:h, :w], 0, 255).astype(np.uint8)
        return img.reshape(shape)

    def as_image_full(self):
        data = self.as_tensor_full()
        img = tensor_to_img(to_cpu((data * 255).squeeze()))
        return img

    def on_resize_ulshift(self, T):
        """Called when the canvas is resized and expands in the upper left direction"""
        # Used by subclass to track corners
        pass

    def extend(self, new_shape, offset=(0, 0)):
        """Dynamically grow size of canvas by allocating new buffer if needed
        and copying over existing canvas. The old canvas is copied into the
        new buffer using `sample_warped_frames` which can handle non-integer shifts
        by using bi-linear interpolation.

        Args:
            new_shape: Shape of new canvas, if too large to fit a new buffer
                is allocated and `resize_count` is incremented.
            offset: Amount by which to shift new canvas w.r.t the origin.

        Returns:
            new_shape, offset
        """
        if np.prod([float(i) for i in new_shape]) >= self.max_pixels:
            new_shape_str = ", ".join(f"{float(i):.2f}" for i in new_shape)
            raise RuntimeError(
                f"Cannot extend canvas to shape ({new_shape_str}) as it will be too large. "
                f"If you still wish to proceed, you can increase the value of `max_pixels`."
            )

        self.origin += to_tensor(offset, device=self.origin.device)
        new_ox, new_oy = self.origin
        real_h, real_w, *_ = self.real_shape
        new_h, new_w, *_ = tuple(new_shape)

        # If new shape is out ou bounds of real shape, expand in that direction by at least
        # `expand_coef` (so coef-1 of padding). Make sure expansion is large enough to fit new shape.
        min_lpad, min_tpad, min_rpad, min_bpad = -new_ox, -new_oy, new_ox + new_w - real_w, new_oy + new_h - real_h
        lr_expansion, td_expansion = (self.expand_coef - 1) * real_w, (self.expand_coef - 1) * real_h

        left_padding = max(lr_expansion, min_lpad + self.min_padding) if min_lpad + self.min_padding > 0 else 0
        right_padding = max(lr_expansion, min_rpad + self.min_padding) if min_rpad + self.min_padding > 0 else 0
        top_padding = max(td_expansion, min_tpad + self.min_padding) if min_tpad + self.min_padding > 0 else 0
        bottom_padding = max(td_expansion, min_bpad + self.min_padding) if min_bpad + self.min_padding > 0 else 0

        real_new_shape = (
            math.ceil(real_h + top_padding + bottom_padding),
            math.ceil(real_w + left_padding + right_padding),
            self.channels,
        )

        # Only resize and copy over data if we need to.
        if real_new_shape != self.real_shape:
            self.resize_count += 1
            new_h, new_w, *_ = real_new_shape
            data = torch.zeros((1, self.channels, new_h, new_w), dtype=self.data.dtype, device=self.data.device)

            if left_padding or top_padding:
                # If the expansion is in the upper/left direction then we need to reset origin.
                padding = torch.tensor([left_padding, top_padding], device=self.origin.device)
                T = translation_matrix(-padding, device=self.origin.device)
                (self.data,) = sample_warped_frames([self.data], T, real_new_shape, device=self.data.device)
                self.origin += padding
                self.on_resize_ulshift(T)
            else:
                # If the origin doesn't move, we can simply allocate a new the buffer
                # and copy over data existing data. This incurs mush less memory overhead than warping.
                data[..., :real_h, :real_w] = self.data
                self.data = data

        self.height, self.width, *_ = [float(i) for i in new_shape]
        self.shape = self.height, self.width, self.channels
        return self.shape, to_tensor(offset, device=self.origin.device)

    def extend_to_frame(self, H, shape):
        """Same as `extend` but computes the `new_shape` and `offset` based on a new frame
        with a bach-warp of `H` and a given shape."""
        return self.extend_to_frames([H], [shape])

    def extend_to_frames(self, Hs, shapes):
        """Multi-frame version of `extend_to_frame`."""
        with torch.no_grad():
            # The "at least one identity warp" assumption is used here
            new_shape, offset = get_max_extent([*Hs, torch.eye(3)], [*shapes, self.shape], device=self.origin.device)
            return self.extend(new_shape, offset)

    def warp_smallest(self, frames, H, shape=None, extra_padding=1, origin=None):
        """Warp frame `y` using homography `H`, but instead of allocating a large buffer
        to warp the frame into, we compute the smallest such buffer and the corresponding
        (x, y) shift that this warp would need to be translated by before adding it to the
        canvas.

        Note: In order to slice at an integer value, we translate the homography by the
            fractional part of the warp offset (plus some padding too). The returned
            warp offset is therefore an integer shift amount.

        Args:
            frames: Frames to warp. Frames are expected to be tensors (NCHW).
            H: Homography that maps the frames *to* the canvas frame.
            shape: Shape of frames as (H,W,C). Inferred if not provided.
            extra_padding: Minimum padding added to the warped frame, should be
                smaller or equal to `min_padding` but at least 1.
            origin: Correct for origin shift by translating by -origin. If None,
                correct by self.origin. Default: None.

        Returns:
            warp_offset, warp_size, warped_frames
        """
        # Convert everything to tensors, and undo origin shift.
        H = to_tensor(H, device=self.origin.device)
        origin = self.origin if origin is None else to_tensor(origin, device=self.origin.device)
        tH = H @ translation_matrix(-origin)

        # Frames are expected to be tensors (NCHW)
        if shape is None:
            if len(shapes := set(tuple(f.shape[-2:]) for f in frames)) != 1:
                raise RuntimeError(f"All frames must have the same shape, instead got shapes {shapes}.")
            h, w = shapes.pop()
        else:
            h, w, *_ = tuple(shape)

        # Get the extent of the new warp, which likely is non-integer, and round it up so it's whole.
        # The extra shift due to the fractional part is accounted for while warping below.
        # TODO: Can we avoid the extra call to `get_max_extent` below? This info is known when we `add_frame`.
        (warp_h, warp_w, *_), warp_offset = get_max_extent([tH], [(h, w)], ceil=True, device=self.origin.device)
        padding = padding_x, padding_y = torch.frac(warp_offset) + extra_padding
        warp_h = int(warp_h + padding_y + extra_padding)
        warp_w = int(warp_w + padding_x + extra_padding)
        warp_offset = (warp_offset - padding).to(int)

        warped_frames = sample_warped_frames(
            frames,
            tH @ translation_matrix(warp_offset),
            (warp_h, warp_w),
            device=self.origin.device,
        )
        return warp_offset, (warp_h, warp_w), warped_frames

    def merge_frame(self, sample, offset, size):
        """Perform the actual merging of a new frame and the canvas.
        Assumes last channel is transparency/alpha mask.

        Implements simple "over" alpha-compositing:
        https://en.wikipedia.org/wiki/Alpha_compositing

        Args:
            sample: new frame to add to canvas
            offset: top left coordinates of new frames in canvas coordinate frame
            size: dimensions of new frame
        """
        start_x, start_y = offset
        warp_h, warp_w, *_ = size
        idxs = np.s_[..., start_y : start_y + warp_h, start_x : start_x + warp_w]

        img, alpha_img = torch.tensor_split(sample, (-1,), dim=1)
        old, alpha_old = torch.tensor_split(self.data[idxs], (-1,), dim=1)

        alpha_new = alpha_img + alpha_old * (1 - alpha_img)
        color_new = (img * alpha_img + old * alpha_old * (1 - alpha_img)) / alpha_new

        self.data[idxs] = torch.nan_to_num(torch.concat([color_new, alpha_new], dim=1))

    def add_frame(self, H, y):
        """Overlay a new frame to the canvas with (back-)projective mapping H.

        Args:
            H: Homography that maps the new frame *to* the canvas
            y: New frame data, will be converted to (1, C, H, W) [0-1] range if needed.

        Returns:
            new_shape, offset
        """
        H = to_tensor(H, device=self.origin.device)
        y = img_to_tensor(y, device=self.origin.device, batched=True)
        *_, c, h, w = tuple(y.shape)

        if c < self.channels - 1:
            raise ValueError(f"Expected frame to have {self.channels-1} or {self.channels} channels, instead got {c}.")
        elif c == self.channels - 1:
            # No alpha channel is present, assume it's all 1's
            y = torch.concat([y, torch.ones(1, 1, h, w, device=y.device)], dim=1)

        new_shape, offset = self.extend_to_frame(H, shape=(h, w))

        # Don't warp image to `real_shape`, instead warp it to the warp's max extent
        # and overlay it on top of `data` via slicing (i.e: a translation) and alpha compositing.
        warp_offset, warp_size, (warped_sample,) = self.warp_smallest([y], H @ translation_matrix(offset), shape=(h, w))
        self.merge_frame(warped_sample, warp_offset, warp_size)

        return new_shape, offset

    def add_frames(self, Hs, ys, shapes=None, pbar=True, online=False, total=None):
        """Similar to `add_frame` but for multiple frames, importantly
        we adjust each H as the canvas grows in size.

        Args:
            Hs: Frame homographies to use (maps frame to canvas)
            ys: Input frames, will be converted to (B, C, H, W) [0-1] range if needed.
            shapes: Shape of all input frames, if not supplied, it will *try* to be inferred at runtime (slower).
            pbar: If true, show progress bar as frames are added. Default: True
            online: If true, process each frame individually instead of in batch. Default: False

        Returns:
            new_shape, offset
        """
        if online:
            offset = torch.tensor([0, 0], dtype=torch.float32, device=self.origin.device)
        else:
            if isinstance(ys, collections.abc.Iterator):
                if shapes is None:
                    # ys might be a generator so it could be too big to list(ys) to find all shapes
                    raise TypeError(
                        "Please provide argument `shapes` when `ys` is an iterator/generator and online is False."
                    )
            elif shapes is None:
                ys = list(ys)
                shapes = [y.shape[-2:] if torch.is_tensor(y) else y.shape[:2] for y in ys]
            Hs = list(Hs)
            new_shape, offset = self.extend_to_frames(Hs, shapes)

        total = total or min(
            getattr(Hs, "__len__", lambda: np.inf)(),
            getattr(ys, "__len__", lambda: np.inf)(),
            getattr(shapes, "__len__", lambda: np.inf)(),
        )
        pbar = tqdm(zip(Hs, ys), total=total if np.isfinite(total) else None) if pbar else zip(Hs, ys)

        for H, y in pbar:
            H_ = to_tensor(H, device=self.origin.device) @ translation_matrix(offset)
            new_shape, new_offset = self.add_frame(H_, y)
            offset += new_offset * int(online)
        return self.shape, offset

    def show(self, full=False):
        plt.imshow(self.as_image() if not full else self.as_image_full(), cmap="gray", vmin=0, vmax=255)
        if full:
            # Imshow makes ticks be in the middle of a cell, here we offset the corners so they bound the cells
            corners = get_corners(translation_matrix(self.origin), self.shape, device=self.origin.device)
            plt.plot(*to_cpu(corners - 0.5)[[0, 1, 2, 3, 0]].T, lw=1, c="b")

    def _repr_png_(self):
        """IPython display hook support"""
        plt.figure(figsize=(18, 8))
        self.show()


class Panorama(DynamicCanvas):
    def __init__(self, shape=(480, 640, 3), expand_coef=1.5, track_corners=False, min_padding=1, **kwargs):
        super().__init__(shape, expand_coef, min_padding, **kwargs)
        self.register_buffer("corners", None)
        self.track_corners = track_corners

    @classmethod
    def construct(cls, *args, levels=1, **kwargs):
        levels = cls._validate_levels(levels)

        if len(levels) != 1:
            raise ValueError("Expected only one level, for multi-level support use the `multi_construct` generator.")

        return next(cls.multi_construct(*args, levels=levels, **kwargs))

    @classmethod
    def multi_construct(
        cls,
        patches,
        levels=1,
        strategy=None,
        burst_size=None,
        device=None,
        wrt=0,
        num_workers=-1,
        workers_per_batch=4,
        endpoints="extrapolate",
        pano_transform="same",
        subbatch_transform=None,
        scale_schedule=1.0,
        ret_pano=True,
        online=False,
        full=False,
        **kwargs,
    ):
        """Construct panorama from input patches using an iterated refinement technique based on
        traditional feature-based matching techniques.

        Args:
            patches: patch generator of type `Patches`/`PatchDataset` or similar.
            levels: level of refinement of final panorama. Expected to be an integer, sequence of integers,
                tuple of (int, bool) pairs, or sequence thereof. The integer corresponds to the number of
                iterations to perform, while the boolean indicated whether to interpolate warps at the last
                level. A level of zero means to perform one iteration but without interpolating the estimated
                warps to all frames and instead using the last iteration's "merged frames" to create the panorama.
                This is equivalent to a value of (1, False). If a list of ints is provided, it will be interpreted
                as (int, True) except if zero. Default: 1
            strategy: sampling strategy to use at each iteration. Must be an iterable where each
                item is a tuple of (midpoints, indices). Default is `Panorama.midpoint_sampler`.
            burst_size: number of frames to merge together at each step. Only used if strategy is None. Default: 50
            device: torch device to use.
            wrt: normalized index (0, 1) of binary frame with identity warp.
            num_workers: number of threads to use while loading in patches, default: -1 (all cores).
            workers_per_batch: workers each batch will be loaded from. See `datasets.BatchedDataLoader`.
            endpoints: if "drop", do not include all frames outside midpoints in final panorama. If "extrapolate"
                all frames will be included and warps will be extrapolated from data.
            scale_schedule: scale of merging operation for each level (starting at level 1). Default: 1.0.
            ret_pano: if false, no panorama will be returned or created, only the merged patches. Default: True
            online: if true, all merged patches will not be loaded into memory (same for weights). A merged patch
                generator will be consumed by the localizer instead of a list of patches. Default: False.
            full: if true yield not only level and panorama but also merged_patches, interp_hs, center_idxs.
            **kwargs: See `_estimate_warps` for more.

        Yields:
            (level_number, interp_last), pano

            if full, also yield:
                center_idxs: indices of frames whose warps were estimated
                merged_patches: patches used to estimate last round of homographies
                interp_hs: the estimated homography interpolant from which the panorama was created
        """
        # Validate arguments
        levels = cls._validate_levels(levels)
        if not all(
            hasattr(patches, attr)
            for attr in ("__getitem__", "__len__", "post_merge_transform", "crop_size", "as_dataset")
        ):
            raise ValueError(
                f"Argument `patches` expected to be of type Patches/PatchDataset or "
                f"similar instead got {type(patches)}."
            )
        if not (0.0 <= wrt <= 1.0):
            raise ValueError(
                f"Argument `wrt` needs to be the normalized index of a binary patch, i.e in [0, 1], got {wrt}."
            )
        if endpoints.lower() not in ("extrapolate", "estimate", "drop"):
            raise ValueError(
                f"Argument `endpoints` expected to be one of extrapolate, estimate, drop but got {endpoints}."
            )
        if not strategy:
            if not burst_size:
                raise ValueError("Argument `burst_size` is required when using default midpoint sampling strategy.")
            strategy = cls.midpoint_sampler(len(patches), burst_size, endpoints=endpoints.lower() == "estimate")
        elif burst_size is not None:
            warnings.warn("Argument `burst_size` will be ignored since a sampling strategy was provided.")
        if isinstance(scale_schedule, float):
            scale_schedule = [scale_schedule] * max(lvl for lvl, _ in levels)
        elif not hasattr(scale_schedule, "__getitem__"):
            raise ValueError("Expected `scale_schedule` argument to be float or subscriptable object returning floats.")
        if not isinstance(pano_transform, list):
            pano_transform = [pano_transform] * max(lvl for lvl, _ in levels)
        for (lvl, interp_last), pano_t in zip(levels, pano_transform):
            if not interp_last and pano_t != "same" and not online:
                raise ValueError(
                    "Cannot apply any other transform to a panorama created using merged frames as "
                    "merged frames will already have been processed using `patches.post_merge_transform`."
                    "Either set `pano_transform` to 'same' for the corresponding level, set online to "
                    "true, or request other levels."
                )

        # Create core warp estimator
        n = len(patches)
        subbatch_transform = subbatch_transform if subbatch_transform is not None else lambda x: x
        num_workers_full = num_workers if num_workers >= 0 else os.cpu_count()
        loader = functools.partial(
            BatchedDataLoader,
            collate_fn=lambda x: x,
            workers_per_batch=workers_per_batch,
            prefetch_factor=1 if num_workers_full > 0 else None,
            num_workers=num_workers_full,
            collate_batch_fn=lambda x: (subbatch_transform(sb) for sb in mitertools.flatten(x)),
        )
        patches_ds = patches.as_dataset()
        estimated_warps = cls._estimate_warps(
            patches_ds, loader, strategy, wrt=wrt, device=device, scale_schedule=scale_schedule, online=online, **kwargs
        )
        estimated_warps = enumerate(estimated_warps, 1)

        def advance_until(target_lvl, data):
            if data is None:
                raise RuntimeError(f"Exhausted iterator before reaching level {target_lvl}!")
            lvl, *_ = data
            return target_lvl == lvl

        for lvl, interp_last in levels:
            # Advance the `estimated_warps` iterator until we've reached current level
            i, warp_data = mitertools.first_true(estimated_warps, pred=functools.partial(advance_until, lvl))

            # Add current data back in front of `estimated_warps` iterator to account for one
            # level being requested multiple times (such as (1, True) and (1, False)).
            estimated_warps = itertools.chain(
                [
                    (i, warp_data),
                ],
                estimated_warps,
            )
            (center_idxs, indices), merged_patches, weights, interp_hs = warp_data

            if ret_pano:
                # Create panorama at (lvl, interp_last) and yield it
                crop_h, crop_w, crop_c = patches_ds.crop_size
                new_size = np.array([crop_h * scale_schedule[lvl - 1], crop_w * scale_schedule[lvl - 1], crop_c])
                pano = cls(new_size.astype(int), **kwargs).to(device)

                if interp_last:
                    # Create dataloader with batch size 1. This allows for multi-threaded
                    # patch retrieval while not loading all patches into memory either.
                    if endpoints.lower() == "drop":
                        patch_idxs = np.arange(np.floor(center_idxs.min(), center_idxs.max())).astype(int)
                    else:
                        patch_idxs = np.arange(n).astype(int)
                    patch_loader = loader(patches_ds, batch_sampler=patch_idxs.reshape(-1, 1))

                    interp_hs = interp_hs.lhs_scale(1 / scale_schedule[lvl - 1])
                    new_shape, offset = pano.add_frames(
                        interp_hs(patch_idxs / (n - 1)),
                        mitertools.flatten(patch_loader),
                        shapes=itertools.repeat(patches_ds.crop_size, len(patch_idxs)),
                        online=False,
                    )

                    # We can treat the whole panorama as a merged_patch, so we need to also apply the
                    # post_merge_transform here in order to match the non-interp_last case
                    if pano_transform[lvl - 1] is not None:
                        pano_img, pano_mask = torch.tensor_split(pano.as_tensor_full(), (-1,), dim=1)

                        if pano_transform[lvl - 1] == "same":
                            pano_img = patches.post_merge_transform(
                                pano_img / pano_mask, level=lvl, burst_size=burst_size, interp_last=interp_last
                            )
                        else:
                            pano_img = pano_transform[lvl - 1](
                                pano_img / pano_mask, level=lvl, burst_size=burst_size, interp_last=interp_last
                            )

                        pano.data[:, : pano.channels - 1, ...] = pano_img * pano_mask
                elif not online:
                    # We have access to the pre-computed merged frames and weights so we use them
                    # Note: Here pano_transforms[lvl-1] should always be "same"
                    merged_patches_ = [torch.concat([mp, w], dim=1) for mp, w in zip(merged_patches, weights)]
                    new_shape, offset = pano.add_frames(interp_hs(center_idxs / (n - 1)), merged_patches_, online=False)
                else:
                    # No merged frames so we need to iterate over all the frames...
                    patch_idxs = np.arange(len(center_idxs) * burst_size).astype(int)
                    patch_loader = loader(patches_ds, batch_sampler=patch_idxs.reshape(-1, 1))
                    interp_hs = interp_hs.lhs_scale(1 / scale_schedule[lvl - 1])

                    new_shape, offset = pano.add_frames(
                        mitertools.repeat_each(interp_hs(center_idxs / (n - 1)), burst_size),
                        mitertools.flatten(patch_loader),
                        shapes=itertools.repeat(patches_ds.crop_size, len(patch_idxs)),
                        online=False,
                        total=len(patch_idxs),
                    )

                    # We can treat the whole panorama as a merged_patch, so we need to also apply the
                    # post_merge_transform here in order to match the non-interp_last case
                    if pano_transform[lvl - 1] is not None:
                        pano_img, pano_mask = torch.tensor_split(pano.as_tensor_full(), (-1,), dim=1)

                        if pano_transform[lvl - 1] == "same":
                            pano_img = patches.post_merge_transform(
                                pano_img / pano_mask, level=lvl, burst_size=burst_size, interp_last=interp_last
                            )
                        else:
                            pano_img = pano_transform[lvl - 1](
                                pano_img / pano_mask, level=lvl, burst_size=burst_size, interp_last=interp_last
                            )

                        pano.data[:, : pano.channels - 1, ...] = pano_img * pano_mask
            else:
                pano = None
                offset = torch.zeros(2)

            out = (lvl, interp_last), pano, offset
            out += (merged_patches, interp_hs, center_idxs) if full else ()
            yield out

    @staticmethod
    def _estimate_warps(
        patches_ds,
        loader,
        sampler,
        localizer=None,
        wrt=0,
        device=None,
        interp_kind="cubic",
        init_hs=None,
        scale_schedule=None,
        crop_merged=False,
        online=True,
        **kwargs,
    ):
        """Core method to estimate warps using an iterated refinement technique based on
        traditional feature-based matching techniques.

        Args:
            patches_ds: patch generator of type `PatchDataset`.
            loader: patch loader which return an iterator when provided with a patch_ds and indices.
            sampler: a sampling strategy such as `midpoint_sampler`, generated burst indices.
            localizer: callable responsible for estimating the warps. It should accept an iterable of merged
                frames and return a list of estimated warps. Defaults to using `homography.compute_homographies`.
            wrt: normalized index (0-1) of binary frame with identity warp.
            device: torch device to use.
            interp_kind: type of interpolation to use for homographies, see scipy's `interp1d`.

        Yields:
            (center_idxs, indices), merged_patches, weights, interp_hs

            Where:
                (center_idxs, indices): item produced by advancing the sampler
                 merged_patches: patches used to estimate last round of homographies
                 weights: merging weights associated with the above `merged_patches`
                 interp_hs: the estimated homography interpolant from which the panorama was created
        """
        # Initialize sampling strategy, homography spline
        n = len(patches_ds)
        scale_gen = iter(scale_schedule)

        if init_hs is None:
            init_hs = [np.eye(3)] * n
        elif len(init_hs) != n:
            raise ValueError("Initial warps specified by `init_hs` expected to have same length as patch generator.")

        if localizer is None:
            localizer = functools.partial(compute_homographies, method="sift", ratio=0.75, startegy="knn")

        interp_hs = homography_interp(init_hs, np.linspace(0, n - 1, n), mode="lk", interp_kind=interp_kind)

        # Main refinement loop
        for lvl, (center_idxs, indices) in enumerate(sampler, 1):
            current_scale = next(scale_gen)

            # Create dataloader to load the patches more efficiently
            patch_loader = loader(patches_ds, batch_sampler=indices)

            # Offload the patch merging to a generator such that if there's way to many groups
            # we can consume them in a streaming fashion instead of loading them all in memory.
            merged_patch_gen = Panorama._get_merged_patches(
                patch_loader,
                lvl,
                center_idxs,
                indices,
                interp_hs,
                current_scale=current_scale,
                device=device,
                crop_merged=crop_merged,
                **kwargs,
            )

            # If `online` then we do not keep merged_patches in memory. The merged_patches generator
            # will be consumed by the localizer, so here we also throw out weights as they will not be used.
            merged_patches, weights, offsets = mitertools.unzip(
                (merged_patch, weight, offset) if not online else (merged_patch, None, offset)
                for merged_patch, weight, offset in merged_patch_gen
            )
            merged_patches, weights, offsets = (list(i) if not online else i for i in (merged_patches, weights, offsets))

            # Estimate homographies between new merged_patches
            burst_hs = localizer((tensor_to_img(p * 255) for p in merged_patches))

            # If `online`, both merged_patches and weights will be empty lists at this point
            merged_patches, weights, offsets = (
                list(filter(lambda x: x is not None, i)) for i in (merged_patches, weights, offsets)
            )

            # Re-project all estimated homographies such that the wrt H is eye
            # Note: We need to interpolate here too as the wrt index isn't necessarily in center_idxs
            wrt_inv = np.linalg.inv(
                homography_interp(burst_hs, center_idxs / (n - 1), mode="lk", interp_kind=interp_kind)(wrt)
            )
            burst_hs = [h @ wrt_inv for h in burst_hs]

            # Offset new estimated homographies by any offsets the merge step created
            burst_hs = [bh @ to_numpy(translation_matrix(*offset)) for bh, offset in zip(burst_hs, offsets)]

            # Create new interpolant for next iteration
            interp_hs = homography_interp(burst_hs, center_idxs / (n - 1), mode="lk", interp_kind=interp_kind)

            yield (center_idxs, indices), merged_patches, weights, interp_hs

    @staticmethod
    def _get_merged_patches(
        patch_loader, lvl, center_idxs, indices, interp_hs, current_scale=1, device=None, crop_merged=False, **kwargs
    ):
        patches_ds = patch_loader.dataset
        n = len(patches_ds)

        # Create a burst frame (merged_patches) using previous iteration's interpolated homographies
        # We need to account for the current warp, so we warp all frames back such that the
        # midpoint frame has identity warping.
        for i, patch_batch in tqdm(enumerate(patch_loader), total=len(center_idxs)):
            crop_h, crop_w, crop_c = patches_ds.crop_size
            new_size = np.array([crop_h * current_scale, crop_w * current_scale, crop_c])
            center_h_inv = np.linalg.inv(interp_hs(center_idxs[i] / (n - 1)))
            patch_hs = [h @ center_h_inv for h in interp_hs(indices[i] / (n - 1))]
            patch_hs = [h @ to_numpy(scale_matrix(1 / current_scale)) for h in patch_hs]

            # Perform main merging of frames
            pano = Panorama(new_size.astype(int), **kwargs).to(device)
            shape, offset = pano.add_frames(
                patch_hs,
                patch_batch,
                shapes=itertools.repeat(patches_ds.crop_size, len(indices[i])),
                online=False,
                pbar=False,
            )

            # Some patch datasets might define an aux merging operation, such as merging masks...
            if hasattr(patches_ds, "auxiliary_merge"):
                patches_ds.auxiliary_merge(patch_hs, device=device)

            img, mask = torch.tensor_split(pano.as_tensor(), (-1,), dim=1)
            merged_patch = patches_ds.post_merge_transform(img / mask, level=lvl, burst_size=len(patch_hs))

            if crop_merged:
                merged_patch, mask = sample_warped_frames(
                    [merged_patch, mask],
                    translation_matrix(-pano.total_shift),
                    patches_ds.crop_size,
                    device=device,
                )

            yield merged_patch.to("cpu"), mask.to("cpu"), offset.to("cpu")

    @staticmethod
    def _validate_levels(
        levels: Union[int, Tuple[int, bool], List[Union[Tuple[int, bool], int]]]
    ) -> List[Tuple[int, bool]]:
        # Convert levels into it's canonical and sorted [(int, bool), ...] form
        if isinstance(levels, int):
            levels = [(levels, bool(levels))]
        elif isinstance(levels, tuple):
            levels = [levels]
        elif isinstance(levels, collections.abc.Iterable):
            levels = [lvl if isinstance(lvl, tuple) else (lvl, bool(lvl)) for lvl in levels]
        else:
            raise ValueError("Argument `levels` is of unexpected type.")

        for lvl, interp_last in levels:
            if lvl < 0:
                raise ValueError(f"Argument `levels` must be greater or equal to 0, got {lvl}.")
            elif lvl == 0 and interp_last:
                # (0, False) is the same as (1, False)...
                raise ValueError("Level (0, True) isn't defined, did you mean 1?")
        return sorted((max(lvl, 1), interp_last) for lvl, interp_last in levels)

    @staticmethod
    def midpoint_sampler(n, burst_size, endpoints=True, strict=False):
        """For a sequence of `n` frames, and given `burst_size`, return indices of burst patch and center index.

        The first batch of indices will exclude ~burst_size/2 frames on either side and be spaced out by burst_size,
        subsequent calls will add indices in between previous ones.

        If the burst size is even, the midpoint will be selected as (burst_size-1)//2. If `n` is not evenly
        divisible by `burst_size`, the remainder is discarded.

        If endpoints is true, the start/end of the sequence of frames will be added with windows that are
        not centered, roughly [:burst_size-1] and [-burst_size:]. However, these will only be added at the
        second iteration as off-center windows do not make sense when frames are just averaged. Subsequent
        iterations will add more points using the same windows.

        Finally, in strict mode, new sequences of midpoints/indices will be generated until new points are
        no longer integers. If not strict, simply round these and stop yielding new sequences when the rounded
        ones start having duplicates.

        Args:
            n: number of frames
            burst_size: averaging window size
            endpoints: if true, include endpoints and their off-center windows
            strict: if true stop as soon as indices are no longer integers,
                other wise round and keep going until duplicates form.

        Returns:
            A generator yielding: midpoints, indices
        """
        num_points = n // burst_size
        left_padd = (burst_size - 1) // 2
        right_padd = (burst_size - 1) - left_padd
        divisible_len = (n // burst_size) * burst_size
        i = 0

        # Stop when the next iteration would create non-integer midpoints
        while ((divisible_len - burst_size) / ((num_points - 1) or 1)).is_integer() or not strict:
            mids = np.linspace(left_padd, divisible_len - 1 - right_padd, num_points)
            mids = np.round(mids).astype(int)

            if len(set(mids)) != len(mids):
                # We've started having duplicates, we cant subdivide further!
                break

            indices = [np.arange(mid - left_padd, mid + right_padd + 1).astype(int) for mid in mids]

            if endpoints and i > 0:
                left_endpoints = np.linspace(0, min(burst_size // 2, n), 2 ** (i - 1), endpoint=False)
                right_endpoints = np.linspace(n - 1, n - 1 - min(burst_size // 2, n), 2 ** (i - 1), endpoint=False)

                mids = np.concatenate([left_endpoints, mids, right_endpoints])
                mids = np.round(mids).astype(int)

                indices = (
                    [np.arange(0, min(burst_size // 2, n))] * len(left_endpoints)
                    + indices
                    + [np.arange(n - 1 - min(burst_size // 2, n), n - 1)] * len(right_endpoints)
                )

            yield mids, indices

            num_points = 2 * num_points - 1
            i += 1

            # Handle degenerate case
            if num_points == 1:
                break

    @property
    def footprint(self):
        """Return estimated memory footprint in bytes"""
        size = super().footprint

        if self.track_corners and self.corners is not None:
            size += self.corners.element_size() * self.corners.nelement()

        return size

    def as_image(self):
        img, mask = torch.tensor_split(self.as_tensor(), (-1,), dim=1)
        img = tensor_to_img(to_cpu((img / mask * 255).squeeze()))
        return np.clip(img, 0, 255).astype(np.uint8)

    def as_image_full(self):
        img, mask = torch.tensor_split(self.as_tensor_full(), (-1,), dim=1)
        img = tensor_to_img(to_cpu((img / mask * 255).squeeze()))
        return np.clip(img, 0, 255).astype(np.uint8)

    def on_resize_ulshift(self, T):
        if self.track_corners and self.corners is not None:
            self.corners = warp_points(torch.inverse(T), self.corners, device=self.corners.device)

    def merge_frame(self, sample, offset, size):
        """Perform the actual merging of a new frame into the panorama.
        Assumes last channel is panorama weights/mask.
        Implements simple linear blending/feathering of frames.

        Args:
            sample: new frame to add to canvas
            offset: top left coordinates of new frames in canvas coordinate frame
            size: dimensions of new frame
        """
        start_x, start_y = offset
        warp_h, warp_w, *_ = size
        new_pano, new_mask = torch.tensor_split(sample, (-1,), dim=1)
        idxs = np.s_[..., start_y : start_y + warp_h, start_x : start_x + warp_w]
        old_pano, old_mask = torch.tensor_split(self.data[idxs], (-1,), dim=1)

        self.data[idxs] = torch.concat(
            [old_pano + torch.nan_to_num(new_pano * new_mask), old_mask + torch.nan_to_num(new_mask)], dim=1
        )

    def add_frame(self, H, y):
        """Add a new frame to the panorama with (back-)projective mapping H.

        Args:
            H: Homography that maps the new frame *to* the panorama
            y: New frame data, will be converted to (1, C, H, W) [0-1] range if needed.

        Returns:
            new_shape, offset
        """
        y = img_to_tensor(y, device=self.origin.device, batched=True)
        *_, c, h, w = tuple(y.shape)

        if c == self.channels - 1:
            y = torch.concat([y, identity_distance_transform((h, w), device=y.device)], dim=1)

        new_shape, offset = super().add_frame(H, y)

        if self.track_corners:
            tH = H @ translation_matrix(offset - self.origin)
            new_corners = get_corners(torch.inverse(tH), (h, w), device=self.origin.device)
            self.corners = new_corners if self.corners is None else torch.cat([self.corners, new_corners])

        return new_shape, offset

    def show(self, full=False, corners=False):
        super().show(full=full)

        if self.track_corners and self.corners is not None and corners:
            if not full:
                points = warp_points(translation_matrix(-self.origin), self.corners, device=self.corners.device)
                points = to_cpu(points)
            else:
                points = to_cpu(self.corners)
            for i in range(4):
                plt.scatter(*points[i::4].T)
