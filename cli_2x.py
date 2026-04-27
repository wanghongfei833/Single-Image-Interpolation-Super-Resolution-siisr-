from __future__ import annotations

import argparse
from pathlib import Path

from siisls.api import Params2x, interpolate_2x_file


def main() -> int:
    p = argparse.ArgumentParser(description="2x interpolation CLI (deployment-friendly)")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--output", default=None, help="Output PNG path (default: <input>_2x.png)")
    p.add_argument("--outdir", default=None, help="Output directory (optional)")
    p.add_argument("--crop", type=int, default=0, help="Center-crop size for faster run; 0 disables")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else None
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / (in_path.stem + "_2x.png")

    params = Params2x()
    written, pval, sval = interpolate_2x_file(in_path, output_path=out_path, params=params, crop=args.crop)
    print(f"Saved: {written}")
    print(f"PSNR={pval:.4f} dB, SSIM={sval:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

