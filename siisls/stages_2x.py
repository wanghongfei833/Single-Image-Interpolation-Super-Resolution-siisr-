from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import (
    ArrayF,
    Phase2x,
    bicubic_interp_2x,
    border_return,
    conv2_same,
    gaussian_kernel,
    image_crop,
    image_extend,
    im2col_sliding_f,
    phase_dsp_2x,
    sub2ind_colmajor,
)


def _topk_indices_desc(values: ArrayF, k: int) -> NDArray[np.int64]:
    """
    Return indices of the top-k largest entries in descending order.
    """
    if k >= values.size:
        idx = np.argsort(values)[::-1]
        return idx.astype(np.int64)
    part = np.argpartition(values, -k)[-k:]
    part = part[np.argsort(values[part])[::-1]]
    return part.astype(np.int64)


def interp_stage1_2x(
    y_lr: ArrayF,
    n: int,
    W: int,
    K: int,
    lam: float,
    sigma: float,
    cw: float,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ArrayF:
    """
    Port of Interp_Stage1_ICASSP2019_2x.m (functional correctness first).

    Args:
        progress_callback: Optional callback(stage_progress: float, message: str)
                          stage_progress in range [0.0, 1.0]
    """
    L = n // 2
    y = image_extend(y_lr, L)
    N_m = y.shape[0] * 2
    N_n = y.shape[1] * 2

    y_pre = bicubic_interp_2x(y)

    ff = gaussian_kernel(9, sigma)
    gg = image_crop(conv2_same(image_extend(y_lr, 9), ff), 9)
    y_pre_lr = bicubic_interp_2x(image_extend(gg, L))

    m1 = (N_m // 2 - L + 1)
    n1 = (N_n // 2 - L + 1)

    # 1-based xx_LR, yy_LR
    ind_total = m1 * n1
    xx_lr_1 = (np.arange(ind_total) % m1) + 1
    yy_lr_1 = (np.arange(ind_total) // m1) + 1
    xx_1 = xx_lr_1 * 2 - 1
    yy_1 = yy_lr_1 * 2 - 1
    ind_2x = sub2ind_colmajor(N_m - n + 1, xx_1.astype(np.int64), yy_1.astype(np.int64))

    max_l = int(np.ceil(((2 * W + 1) ** 2) / 4))
    hr_patch_hat = np.zeros((n * n, ind_total), dtype=np.float64)

    hr_patch = im2col_sliding_f(y_pre, n).astype(np.float64, copy=False)
    hr_patch_lp = im2col_sliding_f(y_pre_lr, n).astype(np.float64, copy=False)
    hr_patch_b_lp_source = hr_patch_lp[:, ind_2x]

    phases = phase_dsp_2x(n)

    # Callback helper
    def report_progress(progress: float, msg: str = "Stage1"):
        if progress_callback is not None:
            progress_callback(progress, msg)

    report_progress(0.0, "Stage1: 处理中...")

    for ind0 in range(ind_total):
        xx_lr = int(xx_lr_1[ind0])
        yy_lr = int(yy_lr_1[ind0])
        xx = xx_lr * 2 - 1
        yy = yy_lr * 2 - 1
        x_min, x_max, y_min, y_max = border_return(xx, yy, N_m, N_n, n, W)

        # Build search coordinates for each phase (1-based), then to 0-based linear indices.
        search_coord_oe = np.zeros((max_l,), dtype=np.int64)
        search_coord_eo = np.zeros((max_l,), dtype=np.int64)
        search_coord_ee = np.zeros((max_l,), dtype=np.int64)

        # OE: y from y_min+1 step2, x from x_min step2
        ys, xs = np.meshgrid(np.arange(y_min + 1, y_max + 1, 2), np.arange(x_min, x_max + 1, 2), indexing="xy")
        s = (ys.ravel() - 1) * (N_m - n + 1) + xs.ravel()
        search_coord_oe[: s.size] = s.astype(np.int64)

        # EO: y from y_min step2, x from x_min+1 step2
        ys, xs = np.meshgrid(np.arange(y_min, y_max + 1, 2), np.arange(x_min + 1, x_max + 1, 2), indexing="xy")
        s = (ys.ravel() - 1) * (N_m - n + 1) + xs.ravel()
        search_coord_eo[: s.size] = s.astype(np.int64)

        # EE: y from y_min+1 step2, x from x_min+1 step2
        ys, xs = np.meshgrid(np.arange(y_min + 1, y_max + 1, 2), np.arange(x_min + 1, x_max + 1, 2), indexing="xy")
        s = (ys.ravel() - 1) * (N_m - n + 1) + xs.ravel()
        search_coord_ee[: s.size] = s.astype(np.int64)

        b_lr = hr_patch_b_lp_source[:, ind0]

        update_oe = _weight_return_stage1(
            b_lr=b_lr,
            hr_patch=hr_patch,
            hr_patch_lp=hr_patch_lp,
            cw=cw,
            search_area_coord=search_coord_oe,
            K=K,
            ind_update=phases.ind_oe,
            ind_oo=phases.ind_oo,
            lam=lam,
        )
        update_eo = _weight_return_stage1(
            b_lr=b_lr,
            hr_patch=hr_patch,
            hr_patch_lp=hr_patch_lp,
            cw=cw,
            search_area_coord=search_coord_eo,
            K=K,
            ind_update=phases.ind_eo,
            ind_oo=phases.ind_oo,
            lam=lam,
        )
        update_ee = _weight_return_stage1(
            b_lr=b_lr,
            hr_patch=hr_patch,
            hr_patch_lp=hr_patch_lp,
            cw=cw,
            search_area_coord=search_coord_ee,
            K=K,
            ind_update=phases.ind_ee,
            ind_oo=phases.ind_oo,
            lam=lam,
        )

        y_patch_hat = np.zeros((n * n,), dtype=np.float64)
        y_patch_hat[phases.ind_oe] = update_oe
        y_patch_hat[phases.ind_eo] = update_eo
        y_patch_hat[phases.ind_ee] = update_ee
        y_patch_hat = np.clip(y_patch_hat, 0.0, 255.0)
        hr_patch_hat[:, ind0] = y_patch_hat

        # Report progress every 1%
        if ind0 % max(1, ind_total // 100) == 0:
            report_progress((ind0 + 1) / ind_total, f"Stage1: {(ind0 + 1) * 100 // ind_total}%")

    # Aggregate patches (overlap-add with uniform weights).
    y_hat = np.zeros((N_m, N_n), dtype=np.float64)
    weight = np.zeros((N_m, N_n), dtype=np.float64)
    for ind0 in range(ind_total):
        xx_lr = int(xx_lr_1[ind0])
        yy_lr = int(yy_lr_1[ind0])
        r0 = (xx_lr * 2 - 1) - 1  # to 0-based
        c0 = (yy_lr * 2 - 1) - 1
        patch = hr_patch_hat[:, ind0].reshape((n, n), order="F")
        y_hat[r0 : r0 + n, c0 : c0 + n] += patch
        weight[r0 : r0 + n, c0 : c0 + n] += 1.0
    y_hat /= weight

    report_progress(1.0, "Stage1: 完成")

    y_pre2 = y_hat
    t = image_crop(y_pre2, n)
    t[0::2, 0::2] = y_lr
    return t


def _weight_return_stage1(
    *,
    b_lr: ArrayF,
    hr_patch: ArrayF,
    hr_patch_lp: ArrayF,
    cw: float,
    search_area_coord: NDArray[np.int64],
    K: int,
    ind_update: NDArray[np.int64],
    ind_oo: NDArray[np.int64],
    lam: float,
) -> ArrayF:
    s_1based = search_area_coord[search_area_coord > 0]
    s0 = (s_1based - 1).astype(np.int64)
    patches = hr_patch_lp[:, s0]
    patches_ori = hr_patch[:, s0]

    ddd = -np.sum(np.abs(b_lr[:, None] - patches), axis=0)
    topk = _topk_indices_desc(ddd, K)
    X = patches[:, topk]
    dddd = np.exp(ddd[topk] / cw)

    Xoo = X[ind_oo, :]
    core = (Xoo.T @ Xoo) + lam * np.diag(dddd[0] / (dddd + 1e-12))
    rhs = Xoo.T @ b_lr[ind_oo]
    coef = np.linalg.solve(core, rhs)

    update = patches_ori[ind_update, :][:, topk] @ coef
    return update


def interp_stage2_2x(
    y_ini: ArrayF,
    n: int,
    W: int,
    K: int,
    lam: float,
    cw: float,
    iters: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[ArrayF, ArrayF]:
    """
    Port of Interp_Stage2_ICASSP2019_2x.m.
    Returns (y_hr, mid_step) where mid_step has shape (H,W,iters).

    Args:
        progress_callback: Optional callback(stage_progress: float, message: str)
                          stage_progress in range [0.0, 1.0] per iteration
    """
    M = y_ini.shape[0] + 2 * n
    N = y_ini.shape[1] + 2 * n
    mid_step = np.zeros((y_ini.shape[0], y_ini.shape[1], iters), dtype=np.float64)
    phases = phase_dsp_2x(n)

    y_pre = image_extend(y_ini, n)

    m1 = (M // 2 - n // 2 + 1)
    n1 = (N // 2 - n // 2 + 1)
    ind_total = m1 * n1

    xx_lr_1 = (np.arange(ind_total) % m1) + 1
    yy_lr_1 = (np.arange(ind_total) // m1) + 1
    xx_1 = xx_lr_1 * 2 - 1
    yy_1 = yy_lr_1 * 2 - 1
    ind_2x = sub2ind_colmajor(M - n + 1, xx_1.astype(np.int64), yy_1.astype(np.int64))

    max_l = int(np.ceil(((2 * W + 1) ** 2) / 4))

    # Callback helper
    def report_progress(progress: float, msg: str = "Stage2"):
        if progress_callback is not None:
            progress_callback(progress, msg)

    for it in range(iters):
        report_progress(it / iters, f"Stage2[{it+1}/{iters}]: 处理中...")

        hr_patch_hat = np.zeros((n * n, ind_total), dtype=np.float64)
        hr_patch = im2col_sliding_f(y_pre, n).astype(np.float64, copy=False)
        hr_patch_b_source = hr_patch[:, ind_2x]

        for ind0 in range(ind_total):
            xx_lr = int(xx_lr_1[ind0])
            yy_lr = int(yy_lr_1[ind0])
            xx = xx_lr * 2 - 1
            yy = yy_lr * 2 - 1
            x_min, x_max, y_min, y_max = border_return(xx, yy, M, N, n, W)

            search_coord_oe = np.zeros((max_l,), dtype=np.int64)
            search_coord_eo = np.zeros((max_l,), dtype=np.int64)
            search_coord_ee = np.zeros((max_l,), dtype=np.int64)

            ys, xs = np.meshgrid(np.arange(y_min + 1, y_max + 1, 2), np.arange(x_min, x_max + 1, 2), indexing="xy")
            s = (ys.ravel() - 1) * (M - n + 1) + xs.ravel()
            search_coord_oe[: s.size] = s.astype(np.int64)

            ys, xs = np.meshgrid(np.arange(y_min, y_max + 1, 2), np.arange(x_min + 1, x_max + 1, 2), indexing="xy")
            s = (ys.ravel() - 1) * (M - n + 1) + xs.ravel()
            search_coord_eo[: s.size] = s.astype(np.int64)

            ys, xs = np.meshgrid(np.arange(y_min + 1, y_max + 1, 2), np.arange(x_min + 1, x_max + 1, 2), indexing="xy")
            s = (ys.ravel() - 1) * (M - n + 1) + xs.ravel()
            search_coord_ee[: s.size] = s.astype(np.int64)

            b = hr_patch_b_source[:, ind0]

            update_oe = _pixel_return_stage2(
                b=b,
                hr_patch=hr_patch,
                cw=cw,
                search_area_coord=search_coord_oe,
                K=K,
                ind_update=phases.ind_oe,
                ind_oo=phases.ind_oo,
                lam=lam,
            )
            update_eo = _pixel_return_stage2(
                b=b,
                hr_patch=hr_patch,
                cw=cw,
                search_area_coord=search_coord_eo,
                K=K,
                ind_update=phases.ind_eo,
                ind_oo=phases.ind_oo,
                lam=lam,
            )
            update_ee = _pixel_return_stage2(
                b=b,
                hr_patch=hr_patch,
                cw=cw,
                search_area_coord=search_coord_ee,
                K=K,
                ind_update=phases.ind_ee,
                ind_oo=phases.ind_oo,
                lam=lam,
            )

            y_patch_hat = np.zeros((n * n,), dtype=np.float64)
            y_patch_hat[phases.ind_oe] = update_oe
            y_patch_hat[phases.ind_eo] = update_eo
            y_patch_hat[phases.ind_ee] = update_ee
            y_patch_hat = np.clip(y_patch_hat, 0.0, 255.0)
            hr_patch_hat[:, ind0] = y_patch_hat

            # Report progress every 5%
            if ind0 % max(1, ind_total // 20) == 0:
                iter_progress = (ind0 + 1) / ind_total
                total_progress = (it + iter_progress) / iters
                report_progress(total_progress, f"Stage2[{it+1}/{iters}]: {int(iter_progress * 100)}%")

        y_hat = np.zeros((M, N), dtype=np.float64)
        weight = np.zeros((M, N), dtype=np.float64)
        for ind0 in range(ind_total):
            r0 = (int(xx_lr_1[ind0]) * 2 - 1) - 1
            c0 = (int(yy_lr_1[ind0]) * 2 - 1) - 1
            patch = hr_patch_hat[:, ind0].reshape((n, n), order="F")
            y_hat[r0 : r0 + n, c0 : c0 + n] += patch
            weight[r0 : r0 + n, c0 : c0 + n] += 1.0
        y_hat /= weight

        y_pre = y_hat
        t = image_crop(y_pre, n)
        t[0::2, 0::2] = y_ini[0::2, 0::2]
        y_pre = image_extend(t, n)
        mid_step[:, :, it] = t

    report_progress(1.0, "Stage2: 完成")
    return t, mid_step


def _pixel_return_stage2(
    *,
    b: ArrayF,
    hr_patch: ArrayF,
    cw: float,
    search_area_coord: NDArray[np.int64],
    K: int,
    ind_update: NDArray[np.int64],
    ind_oo: NDArray[np.int64],
    lam: float,
) -> ArrayF:
    s_1based = search_area_coord[search_area_coord > 0]
    s0 = (s_1based - 1).astype(np.int64)
    patches = hr_patch[:, s0]

    ddd = -np.sum(np.abs(b[:, None] - patches), axis=0)
    topk = _topk_indices_desc(ddd, K)
    X = patches[:, topk]
    dddd = np.exp(ddd[topk] / cw)

    Xoo = X[ind_oo, :]
    core = (Xoo.T @ Xoo) + lam * np.diag(dddd[0] / (dddd + 1e-12))
    rhs = Xoo.T @ b[ind_oo]
    coef = np.linalg.solve(core, rhs)
    update = X[ind_update, :] @ coef
    return update


def interp_stage3_2x(
    y_ini: ArrayF,
    n: int,
    W: int,
    K: int,
    lam: float,
    cw: float,
    iters: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[ArrayF, ArrayF]:
    """
    Port of Interp_Stage3_ICASSP2019_2x.m (note: MATLAB file's function name mismatch).
    Returns (y_hr, mid_step).

    Args:
        progress_callback: Optional callback(stage_progress: float, message: str)
    """
    M = y_ini.shape[0] + 2 * n
    N = y_ini.shape[1] + 2 * n
    mid_step = np.zeros((y_ini.shape[0], y_ini.shape[1], iters), dtype=np.float64)
    phases = phase_dsp_2x(n)

    y_pre = image_extend(y_ini, n)

    m1 = (M // 2 - n // 2 + 1)
    n1 = (N // 2 - n // 2 + 1)
    ind_total = m1 * n1

    xx_lr_1 = (np.arange(ind_total) % m1) + 1
    yy_lr_1 = (np.arange(ind_total) // m1) + 1
    xx_1 = xx_lr_1 * 2 - 1
    yy_1 = yy_lr_1 * 2 - 1
    ind_2x = sub2ind_colmajor(M - n + 1, xx_1.astype(np.int64), yy_1.astype(np.int64))

    max_l = int(np.ceil(((2 * W + 1) ** 2) / 4))

    # Callback helper
    def report_progress(progress: float, msg: str = "Stage3"):
        if progress_callback is not None:
            progress_callback(progress, msg)

    for it in range(iters):
        report_progress(it / iters, f"Stage3[{it+1}/{iters}]: 处理中...")

        hr_patch_hat = np.zeros((n * n, ind_total), dtype=np.float64)
        hr_patch = im2col_sliding_f(y_pre, n).astype(np.float64, copy=False)
        hr_patch_b_source = hr_patch[:, ind_2x]

        for ind0 in range(ind_total):
            xx_lr = int(xx_lr_1[ind0])
            yy_lr = int(yy_lr_1[ind0])
            xx = xx_lr * 2 - 1
            yy = yy_lr * 2 - 1
            x_min, x_max, y_min, y_max = border_return(xx, yy, M, N, n, W)

            search_coord_oe = np.zeros((max_l,), dtype=np.int64)
            search_coord_eo = np.zeros((max_l,), dtype=np.int64)
            search_coord_ee = np.zeros((max_l,), dtype=np.int64)

            ys, xs = np.meshgrid(np.arange(y_min + 1, y_max + 1, 2), np.arange(x_min, x_max + 1, 2), indexing="xy")
            s = (ys.ravel() - 1) * (M - n + 1) + xs.ravel()
            search_coord_oe[: s.size] = s.astype(np.int64)

            ys, xs = np.meshgrid(np.arange(y_min, y_max + 1, 2), np.arange(x_min + 1, x_max + 1, 2), indexing="xy")
            s = (ys.ravel() - 1) * (M - n + 1) + xs.ravel()
            search_coord_eo[: s.size] = s.astype(np.int64)

            ys, xs = np.meshgrid(np.arange(y_min + 1, y_max + 1, 2), np.arange(x_min + 1, x_max + 1, 2), indexing="xy")
            s = (ys.ravel() - 1) * (M - n + 1) + xs.ravel()
            search_coord_ee[: s.size] = s.astype(np.int64)

            b = hr_patch_b_source[:, ind0]

            # Stage3 uses all pixels (Core = X'X + ...) rather than only IND_OO.
            update_oe = _pixel_return_stage3(
                b=b, hr_patch=hr_patch, cw=cw, search_area_coord=search_coord_oe, K=K, ind_update=phases.ind_oe, lam=lam
            )
            update_eo = _pixel_return_stage3(
                b=b, hr_patch=hr_patch, cw=cw, search_area_coord=search_coord_eo, K=K, ind_update=phases.ind_eo, lam=lam
            )
            update_ee = _pixel_return_stage3(
                b=b, hr_patch=hr_patch, cw=cw, search_area_coord=search_coord_ee, K=K, ind_update=phases.ind_ee, lam=lam
            )

            y_patch_hat = np.zeros((n * n,), dtype=np.float64)
            y_patch_hat[phases.ind_oe] = update_oe
            y_patch_hat[phases.ind_eo] = update_eo
            y_patch_hat[phases.ind_ee] = update_ee
            y_patch_hat = np.clip(y_patch_hat, 0.0, 255.0)
            hr_patch_hat[:, ind0] = y_patch_hat

            # Report progress every 5%
            if ind0 % max(1, ind_total // 20) == 0:
                iter_progress = (ind0 + 1) / ind_total
                total_progress = (it + iter_progress) / iters
                report_progress(total_progress, f"Stage3[{it+1}/{iters}]: {int(iter_progress * 100)}%")

        y_hat = np.zeros((M, N), dtype=np.float64)
        weight = np.zeros((M, N), dtype=np.float64)
        for ind0 in range(ind_total):
            r0 = (int(xx_lr_1[ind0]) * 2 - 1) - 1
            c0 = (int(yy_lr_1[ind0]) * 2 - 1) - 1
            patch = hr_patch_hat[:, ind0].reshape((n, n), order="F")
            y_hat[r0 : r0 + n, c0 : c0 + n] += patch
            weight[r0 : r0 + n, c0 : c0 + n] += 1.0
        y_hat /= weight

        y_pre = y_hat
        t = image_crop(y_pre, n)
        t[0::2, 0::2] = y_ini[0::2, 0::2]
        y_pre = image_extend(t, n)
        mid_step[:, :, it] = t

    report_progress(1.0, "Stage3: 完成")
    return t, mid_step


def _pixel_return_stage3(
    *,
    b: ArrayF,
    hr_patch: ArrayF,
    cw: float,
    search_area_coord: NDArray[np.int64],
    K: int,
    ind_update: NDArray[np.int64],
    lam: float,
) -> ArrayF:
    s_1based = search_area_coord[search_area_coord > 0]
    s0 = (s_1based - 1).astype(np.int64)
    patches = hr_patch[:, s0]

    ddd = -np.sum(np.abs(b[:, None] - patches), axis=0)
    topk = _topk_indices_desc(ddd, K)
    X = patches[:, topk]
    dddd = np.exp(ddd[topk] / cw)

    core = (X.T @ X) + lam * np.diag(dddd[0] / (dddd + 1e-12))
    rhs = X.T @ b
    coef = np.linalg.solve(core, rhs)
    update = X[ind_update, :] @ coef
    return update


def interp_stage4_2x(
    y_ini: ArrayF,
    n: int,
    W: int,
    K: int,
    lam: float,
    iters: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[ArrayF, ArrayF]:
    """
    Port of Interp_Stage4_ICASSP2019_2x.m.
    Returns (y_hr, mid_step).

    Args:
        progress_callback: Optional callback(stage_progress: float, message: str)
    """
    y_pre = image_extend(y_ini, n) - 128.0
    M, N = y_pre.shape
    mid_step = np.zeros((y_ini.shape[0], y_ini.shape[1], iters), dtype=np.float64)

    total_pixels = (M - n + 1) * (N - n + 1)

    # Callback helper
    def report_progress(progress: float, msg: str = "Stage4"):
        if progress_callback is not None:
            progress_callback(progress, msg)

    for it in range(iters):
        report_progress(it / iters, f"Stage4[{it+1}/{iters}]: 处理中...")

        hr_patch_hat = np.zeros((n * n, total_pixels), dtype=np.float64)
        hr_patch = im2col_sliding_f(y_pre, n).astype(np.float64, copy=False)

        # Normalize columns to unit L2 (MATLAB normalize(...,'norm',2)).
        norms = np.linalg.norm(hr_patch, axis=0, keepdims=True) + 1e-12
        hr_patch_norm = hr_patch / norms

        for ind0 in range(total_pixels):
            # 1-based xx,yy from linear index in [M-n+1, N-n+1]
            xx = (ind0 % (M - n + 1)) + 1
            yy = (ind0 // (M - n + 1)) + 1
            x_min, x_max, y_min, y_max = border_return(xx, yy, M, N, n, W)

            ys, xs = np.meshgrid(np.arange(y_min, y_max + 1, 1), np.arange(x_min, x_max + 1, 1), indexing="xy")
            s_1based = (ys.ravel() - 1) * (M - n + 1) + xs.ravel()
            s0 = (s_1based - 1).astype(np.int64)

            b = hr_patch[:, ind0]
            corr = b @ hr_patch_norm[:, s0]
            topk = _topk_indices_desc(corr, 4 * K)
            sel = s0[topk]
            dddd = corr[topk]
            X = hr_patch[:, sel]

            core = (X.T @ X) + lam * np.diag(dddd[0] / (dddd + 1e-5))
            rhs = X.T @ b
            coef = np.linalg.solve(core, rhs)
            y_patch = X @ coef
            y_patch = np.clip(y_patch, -128.0, 127.0)
            hr_patch_hat[:, ind0] = y_patch

            # Report progress every 2%
            if ind0 % max(1, total_pixels // 50) == 0:
                iter_progress = (ind0 + 1) / total_pixels
                total_progress = (it + iter_progress) / iters
                report_progress(total_progress, f"Stage4[{it+1}/{iters}]: {int(iter_progress * 100)}%")

        y_hat = np.zeros((M, N), dtype=np.float64)
        weight = np.zeros((M, N), dtype=np.float64)
        for ind0 in range(total_pixels):
            r0 = (ind0 % (M - n + 1))
            c0 = (ind0 // (M - n + 1))
            patch = hr_patch_hat[:, ind0].reshape((n, n), order="F")
            y_hat[r0 : r0 + n, c0 : c0 + n] += patch
            weight[r0 : r0 + n, c0 : c0 + n] += 1.0
        y_hat /= weight

        y_pre = y_hat
        t = image_crop(y_pre, n) + 128.0
        t[0::2, 0::2] = y_ini[0::2, 0::2]
        y_pre = image_extend(t, n) - 128.0
        mid_step[:, :, it] = t

    report_progress(1.0, "Stage4: 完成")
    return t, mid_step
