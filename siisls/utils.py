from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline
from scipy.signal import convolve2d


ArrayF = NDArray[np.floating]


def to_gray_float(img: NDArray) -> ArrayF:
    """
    Convert an image to grayscale float64 in [0, 255] (like MATLAB double(imread)).
    """
    x = np.asarray(img)
    if x.ndim == 2:
        out = x.astype(np.float64, copy=False)
    elif x.ndim == 3 and x.shape[2] in (3, 4):
        # Match MATLAB rgb2gray behavior closely (ITU-R BT.601).
        rgb = x[..., :3].astype(np.float64, copy=False)
        out = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image shape {x.shape}")
    return out


def image_extend(im: ArrayF, L: int) -> ArrayF:
    """
    Mirror-extend like the repo's MATLAB Image_Extend.m.
    """
    if L <= 0:
        return np.array(im, copy=True)
    im = np.asarray(im, dtype=np.float64)
    # The MATLAB code extends by reflecting inside the padded canvas.
    return np.pad(im, ((L, L), (L, L)), mode="reflect")


def image_crop(im: ArrayF, L: int) -> ArrayF:
    if L <= 0:
        return np.array(im, copy=True)
    return np.asarray(im, dtype=np.float64)[L:-L, L:-L]


def border_return(xx: int, yy: int, M: int, N: int, n: int, W: int) -> Tuple[int, int, int, int]:
    """
    Port of Border_Return.m (1-based coordinates).
    Returns (x_min, x_max, y_min, y_max) also 1-based.
    """
    x_min = xx - W
    x_max = xx + W
    y_min = yy - W
    y_max = yy + W

    if xx - W <= 0:
        x_min = 1
        x_max = 1 + 2 * W
    if xx + W > M - n + 1:
        x_min = (M - n + 1) - 2 * W
        x_max = (M - n + 1)
    if yy - W <= 0:
        y_min = 1
        y_max = 1 + 2 * W
    if yy + W > N - n + 1:
        y_max = (N - n + 1)
        y_min = (N - n + 1) - 2 * W
    return x_min, x_max, y_min, y_max


def gaussian_kernel(size: int, sigma: float) -> ArrayF:
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd (match fspecial).")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def bicubic_interp_2x(y: ArrayF) -> ArrayF:
    """
    MATLAB-equivalent of bicubic_interp.m using bicubic interp2.
    Input y is sampled at grid (1:2:M, 1:2:N) in the output.
    """
    y = np.asarray(y, dtype=np.float64)
    m_in, n_in = y.shape
    m_out, n_out = m_in * 2, n_in * 2

    # MATLAB uses meshgrid for positions (1-based).
    # Use splines on a regular grid.
    x_in = np.arange(1, m_out + 1, 2, dtype=np.float64)
    y_in = np.arange(1, n_out + 1, 2, dtype=np.float64)
    x_out = np.arange(1, m_out + 1, 1, dtype=np.float64)
    y_out = np.arange(1, n_out + 1, 1, dtype=np.float64)

    # RectBivariateSpline expects increasing 1D coordinates for each axis.
    spline = RectBivariateSpline(x_in, y_in, y, kx=3, ky=3)
    out = spline(x_out, y_out)

    # Match MATLAB post-fix:
    out[:, -1] = out[:, -2]
    out[-1, :] = out[-2, :]
    return out


@dataclass(frozen=True)
class Phase2x:
    ind_oo: NDArray[np.int64]
    ind_eo: NDArray[np.int64]
    ind_oe: NDArray[np.int64]
    ind_ee: NDArray[np.int64]


def phase_dsp_2x(n: int) -> Phase2x:
    """
    Equivalent of phase_dsp_2x.m but returns 0-based flat indices into an n*n patch
    flattened in MATLAB/Fortran order (column-major).
    """
    rr = np.arange(n)
    cc = np.arange(n)
    R, C = np.meshgrid(rr, cc, indexing="ij")

    # MATLAB uses 1:2:end etc -> 0-based is 0::2 (odd indices in 1-based).
    oo = (R % 2 == 0) & (C % 2 == 0)
    ee = (R % 2 == 1) & (C % 2 == 1)
    eo = (R % 2 == 1) & (C % 2 == 0)
    oe = (R % 2 == 0) & (C % 2 == 1)

    flat = lambda m: np.flatnonzero(m.reshape(-1, order="F")).astype(np.int64)
    return Phase2x(ind_oo=flat(oo), ind_eo=flat(eo), ind_oe=flat(oe), ind_ee=flat(ee))


def im2col_sliding_f(im: ArrayF, n: int) -> ArrayF:
    """
    MATLAB im2col(im, [n n], 'sliding') equivalent.
    Returns shape (n*n, (M-n+1)*(N-n+1)).

    - Patch vectorization is column-major (Fortran order).
    - Patch ordering is (row, col) with row varying fastest, matching sub2ind([M-n+1, N-n+1], row, col).
    """
    im = np.asarray(im, dtype=np.float64)
    M, N = im.shape
    out_cols = (M - n + 1) * (N - n + 1)
    out = np.empty((n * n, out_cols), dtype=np.float64)
    k = 0
    for c in range(N - n + 1):
        for r in range(M - n + 1):
            patch = im[r : r + n, c : c + n]
            out[:, k] = patch.reshape(-1, order="F")
            k += 1
    return out


def sub2ind_colmajor(m: int, row_1based: NDArray[np.int64], col_1based: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    MATLAB sub2ind([m, n], row, col) for column-major linear indexing (1-based inputs).
    Returns 0-based indices suitable for NumPy column vectors where columns are ordered by col, then row.
    """
    return (row_1based - 1) + (col_1based - 1) * m


def psnr(x: ArrayF, y: ArrayF) -> float:
    err = (np.asarray(x, dtype=np.float64).ravel() - np.asarray(y, dtype=np.float64).ravel())
    mse = np.mean(err * err)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim(x: ArrayF, y: ArrayF) -> float:
    # Use skimage's reference implementation.
    from skimage.metrics import structural_similarity as _ssim

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    val = _ssim(x, y, data_range=255.0)
    return float(val)


def conv2_same(im: ArrayF, kernel: ArrayF) -> ArrayF:
    """
    2D convolution, output same size as input (like imfilter default with 'same' for 2D).
    im is already extended when needed in the MATLAB code, so boundary choice is less critical here.
    """
    im = np.asarray(im, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    return convolve2d(im, kernel, mode="same")

