from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from .pipeline_2x import icassp2019_2x
from .utils import ArrayF, psnr, ssim, to_gray_float


@dataclass(frozen=True)
class Params2x:
    # Defaults follow main_2x.m
    n1: int = 8
    n2: int = 6
    W1: int = 20
    W2: int = 20
    W3: int = 30
    W4: int = 30
    K: int = 12
    cw: float = 500.0
    sigma: float = 0.85
    iter1: int = 2
    iter2: int = 2
    iter3: int = 6

    @property
    def n(self) -> Tuple[int, int]:
        return (self.n1, self.n2)

    @property
    def W(self) -> Tuple[int, int, int, int]:
        return (self.W1, self.W2, self.W3, self.W4)

    @property
    def lam(self) -> Tuple[float, float, float, float]:
        n1, n2 = self.n
        return (8 * n1 * n1, 8 * n2 * n2, 8 * n2 * n2, 10 * n2 * n2)


def _center_crop(y: ArrayF, crop: int) -> ArrayF:
    if not crop or crop <= 0:
        return y
    h, w = y.shape
    cs = int(crop)
    cs = min(cs, h, w)
    r0 = (h - cs) // 2
    c0 = (w - cs) // 2
    return y[r0 : r0 + cs, c0 : c0 + cs]


def interpolate_2x_array(
    img: NDArray,
    *,
    params: Optional[Params2x] = None,
    crop: int = 0,
    return_intermediates: bool = False,
    color_mode: str = "rgb",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Union[ArrayF, Tuple[ArrayF, Dict[str, Any]]]:
    """
    Raspberry-Pi-friendly entrypoint.

    Args:
        img: HxW (grayscale) or HxWx3/4 (RGB/RGBA), any dtype.
        params: Algorithm parameters.
        crop: Center crop pixels.
        return_intermediates: Return intermediate results.
        color_mode: "rgb" for color (processes each channel separately),
                    "gray" for grayscale processing.
        progress_callback: Optional callback(progress: float, message: str)
                         progress in range [0.0, 1.0]

    Returns:
        yHR4 in float64 [0,255] with shape == input shape (HxW or HxWx3).
    """
    params = params or Params2x()

    # Determine if input is grayscale or color
    is_color = img.ndim == 3 and img.shape[2] in (3, 4)
    original_shape = img.shape

    print(f"[DEBUG] 输入: shape={img.shape}, is_color={is_color}, color_mode={color_mode}, crop={crop}")

    if color_mode == "gray" or not is_color:
        # Grayscale processing
        y = to_gray_float(img)
        print(f"[DEBUG] 灰度转换后: y.shape={y.shape}")
        y = _center_crop(y, crop)
        print(f"[DEBUG] 裁剪后: y.shape={y.shape}, crop={crop}")

        # 输入已经是 LR 图像，直接进行 2x 超分
        # size_final 是输入尺寸的 2 倍
        size_final = (y.shape[0] * 2, y.shape[1] * 2)
        print(f"[DEBUG] 直接对输入进行 2x 超分，size_final={size_final}")
        y4, y3, y2, y1, mid2, mid3, mid4 = icassp2019_2x(
            y_lr=y,
            size_final=size_final,
            n=params.n,
            W=params.W,
            K=params.K,
            lam=params.lam,
            sigma=params.sigma,
            cw=params.cw,
            iter1=params.iter1,
            iter2=params.iter2,
            iter3=params.iter3,
            progress_callback=progress_callback,
        )
        print(f"[DEBUG] 算法输出后: y4.shape={y4.shape}")

        y4 = np.clip(y4, 0.0, 255.0)
        if not return_intermediates:
            print(f"[DEBUG] 返回: y4.shape={y4.shape}")
            return y4

        info: Dict[str, Any] = {
            "y_hr1": y1,
            "y_hr2": y2,
            "y_hr3": y3,
            "mid_step2": mid2,
            "mid_step3": mid3,
            "mid_step4": mid4,
            "psnr": psnr(y4, y),
            "ssim": ssim(y4, y),
            "input_gray": y,
            "params": params,
            "color": False,
        }
        return y4, info

    else:
        # RGB color processing - process each channel separately
        H, W = img.shape[:2]
        H_out, W_out = H * 2, W * 2
        print(f"[DEBUG] RGB处理: H={H}, W={W}, 期望输出: {H_out}x{W_out}")

        # Extract channels
        if img.shape[2] == 4:
            # RGBA - use RGB only
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img

        # Normalize to float [0, 255]
        img_float = img_rgb.astype(np.float64)
        if img_float.max() <= 1.0:
            img_float *= 255.0

        r_channel = img_float[:, :, 0]
        g_channel = img_float[:, :, 1]
        b_channel = img_float[:, :, 2]

        total_channels = 3
        results = []

        def make_channel_callback(channel_idx: int, total: int):
            """Create callback that reports progress for specific channel."""
            def callback(progress: float, msg: str):
                if progress_callback is not None:
                    # Overall progress: each channel gets 1/3 of the total
                    overall_progress = (channel_idx + progress) / total
                    progress_callback(overall_progress, f"处理{['R','G','B'][channel_idx]}通道: {msg}")
            return callback

        # Process R channel - 输入已经是 LR，直接进行 2x 超分
        y_r = to_gray_float(r_channel)
        y_r = _center_crop(y_r, crop) if crop > 0 else y_r
        size_final_r = (y_r.shape[0] * 2, y_r.shape[1] * 2)
        print(f"[DEBUG] R通道: 输入={y_r.shape}, 期望输出={size_final_r}")
        y_r_hr, _, _, _, _, _, _ = icassp2019_2x(
            y_lr=y_r,
            size_final=size_final_r,
            n=params.n,
            W=params.W,
            K=params.K,
            lam=params.lam,
            sigma=params.sigma,
            cw=params.cw,
            iter1=params.iter1,
            iter2=params.iter2,
            iter3=params.iter3,
            progress_callback=make_channel_callback(0, total_channels),
        )
        results.append(np.clip(y_r_hr, 0.0, 255.0))

        # Process G channel - 输入已经是 LR，直接进行 2x 超分
        y_g = to_gray_float(g_channel)
        y_g = _center_crop(y_g, crop) if crop > 0 else y_g
        size_final_g = (y_g.shape[0] * 2, y_g.shape[1] * 2)
        print(f"[DEBUG] G通道: 输入={y_g.shape}, 期望输出={size_final_g}")
        y_g_hr, _, _, _, _, _, _ = icassp2019_2x(
            y_lr=y_g,
            size_final=size_final_g,
            n=params.n,
            W=params.W,
            K=params.K,
            lam=params.lam,
            sigma=params.sigma,
            cw=params.cw,
            iter1=params.iter1,
            iter2=params.iter2,
            iter3=params.iter3,
            progress_callback=make_channel_callback(1, total_channels),
        )
        results.append(np.clip(y_g_hr, 0.0, 255.0))

        # Process B channel - 输入已经是 LR，直接进行 2x 超分
        y_b = to_gray_float(b_channel)
        y_b = _center_crop(y_b, crop) if crop > 0 else y_b
        size_final_b = (y_b.shape[0] * 2, y_b.shape[1] * 2)
        print(f"[DEBUG] B通道: 输入={y_b.shape}, 期望输出={size_final_b}")
        y_b_hr, _, _, _, _, _, _ = icassp2019_2x(
            y_lr=y_b,
            size_final=size_final_b,
            n=params.n,
            W=params.W,
            K=params.K,
            lam=params.lam,
            sigma=params.sigma,
            cw=params.cw,
            iter1=params.iter1,
            iter2=params.iter2,
            iter3=params.iter3,
            progress_callback=make_channel_callback(2, total_channels),
        )
        results.append(np.clip(y_b_hr, 0.0, 255.0))

        # Stack channels back to RGB
        y4_rgb = np.stack(results, axis=-1)
        print(f"[DEBUG] RGB通道合并后: y4_rgb.shape={y4_rgb.shape}")

        if not return_intermediates:
            if progress_callback is not None:
                progress_callback(1.0, "完成")
            print(f"[DEBUG] RGB返回: y4_rgb.shape={y4_rgb.shape}")
            return y4_rgb

        # Calculate metrics on grayscale version for comparison
        y_gray = to_gray_float(img_rgb)
        y_gray_crop = _center_crop(y_gray, crop) if crop > 0 else y_gray
        size_final_gray = (y_gray_crop.shape[0] * 2, y_gray_crop.shape[1] * 2)
        y4_gray = icassp2019_2x(
            y_lr=y_gray_crop,
            size_final=size_final_gray,
            n=params.n,
            W=params.W,
            K=params.K,
            lam=params.lam,
            sigma=params.sigma,
            cw=params.cw,
            iter1=params.iter1,
            iter2=params.iter2,
            iter3=params.iter3,
        )[0]

        info: Dict[str, Any] = {
            "y_hr": y4_rgb,
            "psnr": psnr(to_gray_float(y4_rgb), to_gray_float(y_gray_crop)),
            "ssim": ssim(to_gray_float(y4_rgb), to_gray_float(y_gray_crop)),
            "params": params,
            "color": True,
        }
        if progress_callback is not None:
            progress_callback(1.0, "完成")
        return y4_rgb, info


PathLike = Union[str, Path]


def interpolate_2x_file(
    input_path: PathLike,
    *,
    output_path: Optional[PathLike] = None,
    params: Optional[Params2x] = None,
    crop: int = 0,
    color_mode: str = "rgb",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Path, float, float]:
    """
    Read an image file, run 2x interpolation, and write PNG.
    Returns (output_path, psnr, ssim) computed vs the input image.

    Args:
        color_mode: "rgb" for color output, "gray" for grayscale.
        progress_callback: Optional callback(progress: float, message: str)
    """
    in_path = Path(input_path)
    img = iio.imread(in_path)
    y4, info = interpolate_2x_array(
        img,
        params=params,
        crop=crop,
        return_intermediates=True,
        color_mode=color_mode,
        progress_callback=progress_callback,
    )

    out_path = Path(output_path) if output_path is not None else in_path.with_suffix("").with_name(in_path.stem + "_2x.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle color output
    if info.get("color", False) and y4.ndim == 3 and y4.shape[2] == 3:
        # RGB output
        output_array = np.clip(y4, 0.0, 255.0).round().astype(np.uint8)
        iio.imwrite(out_path, output_array)
    else:
        # Grayscale output
        output_array = np.clip(y4, 0.0, 255.0).round().astype(np.uint8)
        if output_array.ndim == 3:
            output_array = output_array[:, :, 0]  # Take first channel if still 3D
        iio.imwrite(out_path, output_array)

    return out_path, float(info["psnr"]), float(info["ssim"])
