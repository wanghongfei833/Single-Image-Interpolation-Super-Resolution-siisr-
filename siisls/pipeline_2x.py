from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .stages_2x import interp_stage1_2x, interp_stage2_2x, interp_stage3_2x, interp_stage4_2x
from .utils import ArrayF


def icassp2019_2x(
    y_lr: ArrayF,
    size_final: Tuple[int, int],
    n: Tuple[int, int],
    W: Tuple[int, int, int, int],
    K: int,
    lam: Tuple[float, float, float, float],
    sigma: float,
    cw: float,
    iter1: int,
    iter2: int,
    iter3: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[ArrayF, ArrayF, ArrayF, ArrayF, ArrayF, ArrayF, ArrayF]:
    """
    Port of icassp2019_2x.m.
    Returns (yHR4, yHR3, yHR2, yHR1, mid_step2, mid_step3, mid_step4).

    Args:
        progress_callback: Optional callback(progress: float, message: str)
                         progress in range [0.0, 1.0] for overall pipeline
    """
    # Stage 1: 0-10%
    def stage1_callback(progress: float, msg: str):
        if progress_callback is not None:
            progress_callback(progress * 0.1, msg)

    y1 = interp_stage1_2x(
        y_lr, n=n[0], W=W[0], K=K, lam=lam[0], sigma=sigma, cw=cw,
        progress_callback=stage1_callback
    )

    # Stage 2: 10-30%
    def stage2_callback(progress: float, msg: str):
        if progress_callback is not None:
            progress_callback(0.1 + progress * 0.2, msg)

    y2, mid2 = interp_stage2_2x(
        y1, n=n[1], W=W[1], K=K, lam=lam[1], cw=cw, iters=iter1,
        progress_callback=stage2_callback
    )

    # Stage 3: 30-55%
    def stage3_callback(progress: float, msg: str):
        if progress_callback is not None:
            progress_callback(0.3 + progress * 0.25, msg)

    y3, mid3 = interp_stage3_2x(
        y2, n=n[1], W=W[2], K=K, lam=lam[2], cw=cw, iters=iter2,
        progress_callback=stage3_callback
    )

    # Stage 4: 55-100%
    def stage4_callback(progress: float, msg: str):
        if progress_callback is not None:
            progress_callback(0.55 + progress * 0.45, msg)

    y4, mid4 = interp_stage4_2x(
        y3, n=n[1], W=W[3], K=K, lam=lam[3], iters=iter3,
        progress_callback=stage4_callback
    )

    H, Wf = size_final
    y1 = y1[:H, :Wf]
    y2 = y2[:H, :Wf]
    y3 = y3[:H, :Wf]
    y4 = y4[:H, :Wf]

    mid2 = mid2[:H, :Wf, :]
    mid3 = mid3[:H, :Wf, :]
    mid4 = mid4[:H, :Wf, :]

    if progress_callback is not None:
        progress_callback(1.0, "完成")

    return y4, y3, y2, y1, mid2, mid3, mid4
