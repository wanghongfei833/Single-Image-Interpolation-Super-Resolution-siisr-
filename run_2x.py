from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from siisls.pipeline_2x import icassp2019_2x
from siisls.utils import psnr, ssim, to_gray_float


def main() -> int:
    p = argparse.ArgumentParser(description="Python port: ICASSP2019 2x single image interpolation")
    p.add_argument("--input", type=str, default=str(Path("..") / "TESTSET" / "ceshi_1.jpg"))
    p.add_argument("--outdir", type=str, default=str(Path("outputs")))
    p.add_argument(
        "--crop",
        type=int,
        default=128,
        help="Optional center-crop size for quick functional verification. Set 0 to disable.",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = iio.imread(in_path)
    y = to_gray_float(img)

    if args.crop and args.crop > 0:
        h, w = y.shape
        cs = int(args.crop)
        cs = min(cs, h, w)
        r0 = (h - cs) // 2
        c0 = (w - cs) // 2
        y = y[r0 : r0 + cs, c0 : c0 + cs]

    y_lr = y[0::2, 0::2]

    # Default hyper-params match main_2x.m
    n = (8, 6)
    W = (20, 20, 30, 30)
    K = 12
    lam = (8 * n[0] ** 2, 8 * n[1] ** 2, 8 * n[1] ** 2, 10 * n[1] ** 2)
    cw = 500.0
    sigma = 0.85
    iter1 = 2
    iter2 = 2
    iter3 = 6

    y4, y3, y2, y1, mid2, mid3, mid4 = icassp2019_2x(
        y_lr=y_lr,
        size_final=y.shape,
        n=n,
        W=W,
        K=K,
        lam=lam,
        sigma=sigma,
        cw=cw,
        iter1=iter1,
        iter2=iter2,
        iter3=iter3,
    )

    y4c = np.clip(y4, 0.0, 255.0)
    pval = psnr(y4c, y)
    sval = ssim(y4c, y)
    print(f"PSNR={pval:.4f} dB, SSIM={sval:.6f}")

    # Save as uint8 PNG
    out_path = outdir / f"{in_path.stem}_2x.png"
    iio.imwrite(out_path, y4c.round().astype(np.uint8))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

