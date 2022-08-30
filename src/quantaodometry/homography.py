import functools
import itertools
import warnings

import networkx as nx
import numpy as np
import shapely
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from networkx.algorithms import approximation as approx
from numpy import linalg as LA
from scipy import linalg as SLA
from scipy.interpolate import interp1d

from .utils import (img_to_tensor, scale_matrix, to_numpy, to_tensor,
                    torch_logm, translation_matrix)


class homography_interp:
    def __init__(self, hs, ts=None, *, mode, interp_kind="cubic"):
        # TODO: Allow for interpolation to be done on GPU by using something
        #  like: https://stackoverflow.com/questions/61616810 ?
        ts = np.linspace(0, 1, len(hs)) if ts is None else ts
        self.ts, self.hs = to_numpy(ts, hs)
        self.interp_kind = interp_kind
        self.mode = mode

        if mode is None:
            self._interp_fn = self.matrix_interp
        elif mode.lower() in ("lk", "lie"):
            h_params = [params_from_homography(h, mode=mode) for h in hs]
            interpolant = interp1d(ts, h_params, kind=interp_kind, axis=0, fill_value="extrapolate")
            self._interp_fn = functools.partial(self.params_interp, interpolant=interpolant, mode=mode)
        else:
            raise ValueError(f"Mode expected to be one of None, 'lk', 'lie'. Instead got {mode}.")

    def rhs_scale(self, scale_factor):
        """Return copy of interpolant with right hand side scaling applied"""
        hs = [h @ to_numpy(scale_matrix(scale_factor)) for h in self.hs]
        return homography_interp(hs, ts=self.ts, mode=self.mode, interp_kind=self.interp_kind)

    def lhs_scale(self, scale_factor):
        """Return copy of interpolant with left hand side scaling applied"""
        hs = [to_numpy(scale_matrix(scale_factor)) @ h for h in self.hs]
        return homography_interp(hs, ts=self.ts, mode=self.mode, interp_kind=self.interp_kind)

    def validate(self, t):
        if t.max() > self.ts.max() or t.min() < self.ts.min():
            raise ValueError("Query point is outside of interpolation range.")

    def matrix_interp(self, t):
        # Perform 'simple' interpolation, effectively H^t.
        # See: https://mruffini.github.io/assets/papers/view-synthesis.pdf
        # The result here might be complex, these shouldn't occur but might, warn if it does.

        t_shape = np.shape(t)
        t = np.atleast_1d(t)
        self.validate(t)

        # Get indices of left and right homographies for each query point t.
        idxs = np.searchsorted(self.ts, t)

        # Re-normalize t within each interval, compute interpolant
        # Note: As of scipy 1.9.3, logm  isn't vectorized (while expm is), so we loop...
        src_t, dst_t = self.ts[idxs - 1], self.ts[idxs]
        t = (t - src_t) / (dst_t - src_t)
        interp = np.array(
            [
                self.hs[i - 1] @ SLA.expm(ti * SLA.logm(LA.inv(self.hs[i - 1]) @ self.hs[i]))
                for ti, i in zip(t, idxs.flatten())
            ]
        )

        if np.iscomplex(interp).any():
            warnings.warn(
                f"Encountered complex homography while interpolating, with max imaginary "
                f"part of {interp.imag.max():2f}. Casting to real valued.",
                RuntimeWarning,
            )
        return interp.real.reshape(*t_shape, 3, 3)

    @staticmethod
    def params_interp(t, interpolant, mode):
        return homography_from_params(interpolant(t), mode=mode)

    def __call__(self, t):
        return self._interp_fn(t)


def homography_from_params(H, *, mode):
    is_tensor = torch.is_tensor(H)
    H = to_tensor(H)

    if mode.lower() == "lie":
        # See: C. Mei, S. Benhimane, E. Malis, and P. Rives,
        #   "Homography-based Tracking for Central Catadioptric Cameras,"
        #   in 2006 IEEE/RSJ International Conference on Intelligent Robots
        #   and Systems, Beijing, China, Oct. 2006, pp. 669-674.
        #   doi: 10.1109/IROS.2006.282553.
        # And: https://ethaneade.com/lie_groups.pdf
        h1, h2, h3, h4, h5, h6, h7, h8 = H.chunk(8, dim=-1)
        A = torch.stack(
            [
                torch.cat([h5, h3, h1], dim=-1),
                torch.cat([h4, -h5 - h6, h2], dim=-1),
                torch.cat([h7, h8, h6], dim=-1),
            ],
            dim=-2,
        ).matrix_exp()
        return to_numpy(A) if not is_tensor else A
    elif mode.lower() == "lk":
        # Alternate Lucas-Kanade like formulation.
        # See: Appendix A of https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf
        h1, h2, h3, h4, h5, h6, h7, h8 = H.chunk(8, dim=-1)
        A = torch.stack(
            [
                torch.cat([h1 + 1, h3, h5], dim=-1),
                torch.cat([h2, h4 + 1, h6], dim=-1),
                torch.cat([h7, h8, torch.ones_like(h1)], dim=-1),
            ],
            dim=-2,
        )
        return to_numpy(A) if not is_tensor else A
    raise ValueError(f"Mode expected to be one of 'lk', 'lie'. Instead got {mode}.")


def params_from_homography(H, *, mode):
    if mode.lower() == "lie":
        if torch.is_tensor(H):
            # TODO: This needs to be re-written using indexing like below to enable back-prop.
            h5, h3, h1, h4, _, h2, h7, h8, h6 = torch_logm(H).flatten()
            return torch.tensor([h1, h2, h3, h4, h5, h6, h7, h8], device=H.device)
        h5, h3, h1, h4, _, h2, h7, h8, h6 = SLA.logm(H).flatten()
        return np.array([h1, h2, h3, h4, h5, h6, h7, h8])
    elif mode.lower() == "lk":
        H = H.flatten() / H[-1, -1]
        H = H[[0, 3, 1, 4, 2, 5, 6, 7]]
        offsets = [1, 0, 0, 1, 0, 0, 0, 0]
        if torch.is_tensor(H):
            return H - torch.tensor(offsets, device=H.device)
        return to_numpy(H) - np.array(offsets)
    raise ValueError(f"Mode expected to be one of 'lk', 'lie'. Instead got {mode}.")


def scale_homographies(hs, scale=1, *, mode):
    if mode.lower() != "lk":
        raise NotImplementedError("Can only scale in 'lk' mode for now.")

    scale = np.array([1, 1, 1, 1, 1 / scale, 1 / scale, scale, scale])
    return [homography_from_params(params_from_homography(to_numpy(h), mode=mode) * scale, mode=mode) for h in hs]


def warp_points(H, points, epsilon=1e-12, device=None):
    H, points = to_tensor(H, points, device=device, dtype=torch.float32)
    points = points if points.ndim >= 2 else points[None]

    *extra_dims_points, n, dims = tuple(points.shape)
    *extra_dims_H, _, _ = tuple(H.shape)

    if extra_dims_points or extra_dims_H:
        raise NotImplementedError(
            "Function `warp_points` is not yet batched. Parameters are expected to"
            "have shape (N, 2) and (3, 3) not (B, N, 2) and (B, 3, 3) or other."
        )

    if torch.allclose(H, torch.eye(3, device=H.device)):
        return points

    if dims == 2:
        points = torch.hstack((points, torch.ones((len(points), 1), dtype=torch.float32, device=device)))
    elif dims != 3:
        raise ValueError("Expected points to be an Nx2 or Nx3 array")

    points = H @ points.T
    return points.T[..., :2] / (points.T[..., -1:] + epsilon)


def rescale_points_to_frames(frames, points):
    # Frames are expected to be tensors (NCHW)
    if len(shapes := set(tuple(f.shape[-2:]) for f in frames)) != 1:
        raise RuntimeError(f"All frames must have the same shape when sampling, instead got shapes {shapes}.")
    h, w = shapes.pop()

    # Rescale points to [-1, 1]
    points = points.to(torch.float32) / torch.tensor([w, h], device=points.device).reshape(1, 1, 1, 2)
    points = 2 * points - 1
    return points


def sample_frames(frames, points, rescale=True):
    if rescale:
        points = rescale_points_to_frames(frames, points)

    # Note: List comp here instead of passing batch to allow for frames with differing channels depths.
    return [torch.nn.functional.grid_sample(f, points, align_corners=False) for f in frames]


def sample_warped_frames(frames, H, size, device=None):
    """Sample frames at all warped points from an image with `size` using homography `H`."""
    total_h, total_w, *_ = size
    points = meshgrid_points(tuple(size), device=device)
    warped_points = warp_points(H, points, device=device).reshape(1, total_h, total_w, 2)
    return sample_frames(frames, warped_points, rescale=True)


def get_corners(T, size, centered=False, **kwargs):
    """Return (clockwise) location of bounding box corners after homography is applied."""
    h, w, *_ = size
    corners = torch.tensor([[0, 0], [w, 0], [w, h], [0, h]], dtype=torch.float32)

    if centered:
        corners -= torch.tensor([[w / 2, h / 2]], dtype=torch.float32)

    corners = warp_points(T, corners, **kwargs)
    return corners


def get_max_extent(Hs, sizes, ceil=False, dtype=torch.float32, device=None):
    if len(Hs) != len(sizes):
        raise ValueError("Parameters `Hs` and `sizes` must be of same length!")

    Hs = [h if torch.is_tensor(h) else torch.tensor(h) for h in Hs]
    Hs_ = [torch.inverse(h.to(dtype=dtype, device=device)) for h in Hs]
    corners = torch.cat([get_corners(h, size, device=device) for h, size in zip(Hs_, sizes)]).reshape(-1, 2)
    w_h = corners.max(dim=0).values - corners.min(dim=0).values
    total_w, total_h = torch.ceil(w_h).to(int) if ceil else w_h
    offset = corners.min(dim=0).values
    return (total_h, total_w), offset


def corners_in_bounds(T, size, bounds):
    """Return true if the bbox falls within the image bounds."""
    ud, lr, *_ = bounds
    u, d = ud if isinstance(ud, (list, tuple)) else (0, ud)
    l, r = lr if isinstance(lr, (list, tuple)) else (0, lr)
    x, y = get_corners(T, size).T
    return (l <= x).all() and (x < r).all() and (u <= y).all() and (y < d).all()


def meshgrid_points(shape, device=None, centered=False):
    h, w, *_ = tuple(shape)
    X, Y = torch.arange(w, device=device), torch.arange(h, device=device)

    # Add 0.5 as to use center of pixels, not upper right corner.
    X, Y = X + 0.5, Y + 0.5

    if centered:
        X, Y = X - w / 2, Y - h / 2
    return torch.cartesian_prod(Y, X).fliplr()


def generate_homographies(
    img_size,
    num_kps=10,
    scale_h=0.001,
    crop_size=(300, 400),
    n=1000,
    interp_kind="cubic",
    include_i=True,
    ordering=False,
    x=None,
    y=None,
):
    """Randomly generate homography matrices.

    We do this by creating `num_kps` keypoint homographies and interpolating
    between them smoothly to create a total of `n` matrices.

    Note: Only keypoint homographies are guaranteed to be in bounds of img_size.

    `img_size`: Size of original image, used to make sure keypoint homographies are within bounds.
    `num_kps`: Number of "key points" homographies to create and interpolate from.
    `scale_h`: Controls how deformed the projections are, should be quite small.
    `crop_size`: Size of each resulting crop.
    `n`: Number of homographies in total
    `interp_kind`: type of parameter interpolation to perform
    `include_i`: If true, the first homography is only a translation
    `ordering`: If true, try to order keypoint homographies using TSP
    """
    img_h, img_w, *_ = img_size
    crop_h, crop_w, *_ = crop_size
    kp_params = []

    if include_i:
        params = np.concatenate(
            [
                [x] if x is not None else np.random.uniform(size=1, low=0, high=img_w),
                [y] if y is not None else np.random.uniform(size=1, low=0, high=img_h),
                [0, 0, 0, 0, 0, 0],
            ]
        )
        kp_params.append(params)

    while len(kp_params) < num_kps:
        params = np.concatenate(
            [
                [x] if x is not None else np.random.uniform(size=1, low=0, high=img_w),
                [y] if y is not None else np.random.uniform(size=1, low=0, high=img_h),
                np.random.normal(size=6, loc=0, scale=scale_h / np.sqrt(img_h**2 + img_w**2)),
            ]
        )
        # Move homography to center of crop instead of origin
        h = homography_from_params(params, mode="lie") @ translation_matrix(-crop_w / 2, -crop_h / 2)

        if corners_in_bounds(h, crop_size, img_size):
            kp_params.append(params)

    if ordering:
        G = nx.Graph()
        dist = lambda a, b: float(np.sqrt(((a[:2] - b[:2]) ** 2).sum()))
        G.add_weighted_edges_from([(tuple(a), tuple(b), dist(a, b)) for a, b in itertools.product(kp_params, kp_params)])
        kp_params = approx.greedy_tsp(G, source=tuple(kp_params[0]))

        # Break up cycle into a path by cutting largest edge
        first_edge = G.edges[kp_params[0], kp_params[1]]["weight"]
        last_edge = G.edges[kp_params[-1], kp_params[-2]]["weight"]
        kp_params = kp_params[:-1] if last_edge > first_edge else kp_params[1:][::-1]

    params = interp1d(np.linspace(0, 1, num_kps), kp_params, kind=interp_kind, axis=0)(np.linspace(0, 1, n))
    kp_hs = np.array(
        [homography_from_params(r, mode="lie") @ translation_matrix(-crop_w / 2, -crop_h / 2) for r in kp_params]
    )
    hs = np.array([homography_from_params(r, mode="lie") @ translation_matrix(-crop_w / 2, -crop_h / 2) for r in params])

    return kp_hs, hs


def generate_homographies_in_bounds(img_size, *args, num_tries=20, crop_size=(300, 400), **kwargs):
    """Same as `generate_homographies`, but if it returns, all bbox are guaranteed to be in bounds."""
    for _ in range(num_tries):
        kp_hs, hs = generate_homographies(img_size, *args, crop_size=crop_size, **kwargs)

        if all(corners_in_bounds(h, crop_size, img_size) for h in hs):
            return kp_hs, hs

    raise RuntimeError(f"Failed to find a set of suitable homographies after {num_tries} tries.")


def animate_homographies(hs, img_size, crop_size=(300, 400)):
    img_h, img_w, *_ = img_size

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(xlim=(0, img_w), ylim=(img_h, 0))
    (line,) = ax.plot([], [])

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        corners = get_corners(hs[i], crop_size)
        line.set_data(*corners[[0, 1, 2, 3, 0]].T)
        return (line,)

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(hs), interval=20, blit=True)
    plt.close(fig)
    return anim


def polygon_sdf_vec(points, vertices, device=None):
    # Adapted from: https://www.shadertoy.com/view/wdBXRW
    d = ((points - vertices[0]) ** 2).sum(dim=-1)
    s = torch.ones(len(points), device=device)

    for i in range(len(vertices)):
        # distance
        e = vertices[i - 1] - vertices[i]
        w = points - vertices[i]
        b = w - e * torch.minimum(
            torch.tensor(1.0), torch.maximum((w * e).sum(dim=-1) / (e @ e), torch.tensor(0.0))
        ).reshape(-1, 1)
        d = torch.minimum(d, (b**2).sum(dim=-1))

        # winding number from http://geomalgorithms.com/a03-_inclusion.html
        cond = torch.stack(
            [
                points[:, 1] >= vertices[i][1],
                points[:, 1] < vertices[i - 1][1],
                e[0] * w[:, 1] > e[1] * w[:, 0],
            ]
        )

        cond = cond.sum(dim=0)
        cond = (cond == 0) | (cond == 3)
        s[cond] = -s[cond]

    return s * torch.sqrt(d)


def polygon_distance_transform(points, vertices, device=None):
    dists = -polygon_sdf_vec(points, vertices, device=device)
    dists[dists < 0] = 0
    return dists


def distance_transform(h, crop_size, size, device=None):
    corners = get_corners(h, crop_size, device=device)
    points = meshgrid_points(size, device=device)
    dists = polygon_distance_transform(points, corners, device=device)
    return dists.reshape(size).squeeze()


@functools.lru_cache(maxsize=32)
def identity_distance_transform(shape, zero_border=False, device=None):
    """Returns non-warped mask of size `shape`

    If `zero_border`, the border of the distance mask will be zero, otherwise the row/cols
    just outside the boundary are assumed to be zero.
    """
    h, w, *_ = tuple(shape)

    if not zero_border:
        h += 2
        w += 2

    mask = distance_transform(torch.eye(3, device=device), (h, w), (h, w), device=device)

    if not zero_border:
        mask = mask[1:-1, 1:-1]

    return mask[None, None] / mask.max()


def warp_patches(img, hs, size, device=None):
    """Apply homography to image, uses reverse mapping and linear interpolation."""
    img = img_to_tensor(img, batched=True, device=device)

    for h in hs:
        yield sample_warped_frames([img], h, size, device=device)[0]


def outline_from_homographies(hs, crop_size):
    return np.stack(shapely.union_all([shapely.Polygon(get_corners(h, crop_size).numpy()) for h in hs]).exterior.xy)
