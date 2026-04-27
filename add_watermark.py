"""
图片水印工具
为图片添加文字水印，防止劳动成果被盗用
支持中文和斜向水印
"""

from PIL import Image, ImageDraw, ImageFont
import os
import math

def add_watermark(input_path, output_path=None, text="Protected", position="bottom-right", diagonal=None):
    """
    添加水印

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径，None则在原文件名前加 _watermarked
        text: 水印文字（支持中文，用\\n分隔多行）
        position: 水印位置 ('bottom-right', 'bottom-left', 'top-right', 'top-left', 'center')
        diagonal: 斜向水印角度，可选 "left" (左上到右下) 或 "right" (左下到右上)
    """
    img = Image.open(input_path).convert("RGBA")
    width, height = img.size

    # 创建水印层
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)

    # 根据图片大小调整字体大小
    font_size = max(int(min(width, height) * 0.04), 20)

    # 加载支持中文的字体
    font = None
    font_paths = [
        "msyh.ttc",      # 微软雅黑
        "simhei.ttf",    # 黑体
        "simsun.ttc",    # 宋体
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            print(f"使用字体: {fp}")
            break
        except:
            continue

    if font is None:
        font = ImageFont.load_default()
        print("警告: 未找到中文字体，使用默认字体")

    # 斜向水印模式
    if diagonal:
        add_diagonal_watermark(txt_layer, width, height, text, font, diagonal)
    else:
        add_normal_watermark(draw, width, height, text, font, position)

    # 合并图层
    watermarked = Image.alpha_composite(img, txt_layer)

    # 转回 RGB 模式保存
    if watermarked.mode == "RGBA":
        watermarked = watermarked.convert("RGB")

    # 确定输出路径
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        suffix = "_diagonal" if diagonal else "_watermarked"
        output_path = f"{name}{suffix}.jpg"

    watermarked.save(output_path, quality=95)
    print(f"水印已添加: {output_path}")
    return output_path


def add_normal_watermark(draw, width, height, text, font, position):
    """普通位置水印"""
    margin = 30

    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_lines = text.split('\n')
    line_height = font.size + 15

    positions = {
        "bottom-right": (width - text_width - margin, height - text_height - margin),
        "bottom-left": (margin, height - text_height - margin),
        "top-right": (width - text_width - margin, margin),
        "top-left": (margin, margin),
        "center": ((width - text_width) // 2, (height - text_height) // 2),
    }

    pos = positions.get(position, positions["bottom-right"])

    # 绘制文字（带阴影）
    shadow_offset = 2
    for i, line in enumerate(text_lines):
        line_y = pos[1] + i * line_height
        # 阴影
        draw.text((pos[0] + shadow_offset, line_y + shadow_offset), line, font=font, fill=(0, 0, 0, 100))
        # 水印文字
        draw.text((pos[0], line_y), line, font=font, fill=(255, 255, 255, 220))


def add_diagonal_watermark(img_layer, width, height, text, font, direction):
    """斜向水印 - 平铺整个图片"""
    # 旋转角度
    if direction == "right":
        angle = -45  # 左下到右上
    else:  # left
        angle = 45   # 左上到右下

    # 创建临时层绘制文字（透明背景）
    temp_layer = Image.new("RGBA", (width * 3, height * 3), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_layer)

    # 计算平铺间距
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 60
    text_h = bbox[3] - bbox[1] + 40

    spacing_x = text_w
    spacing_y = text_h

    # 平铺文字
    for y in range(0, temp_layer.height, spacing_y):
        for x in range(0, temp_layer.width, spacing_x):
            # 阴影
            temp_draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 60))
            # 白色文字（半透明）
            temp_draw.text((x, y), text, font=font, fill=(255, 255, 255, 100))

    # 旋转
    rotated = temp_layer.rotate(angle, resample=Image.BILINEAR, center=(temp_layer.width // 2, temp_layer.height // 2))

    # 裁剪回原尺寸
    left = (rotated.width - width) // 2
    top = (rotated.height - height) // 2
    rotated = rotated.crop((left, top, left + width, top + height))

    # 合并到水印层
    img_layer.paste(rotated, (0, 0), rotated)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="添加图片水印")
    parser.add_argument("--input", "-i", default="jietu.png", help="输入图片路径")
    parser.add_argument("--output", "-o", default=None, help="输出图片路径")
    parser.add_argument("--text", "-t", default="版权所有 禁止盗用", help="水印文字（用\\n分隔多行）")
    parser.add_argument("--pos", "-p", default="bottom-right",
                        choices=["bottom-right", "bottom-left", "top-right", "top-left", "center"],
                        help="水印位置（不使用斜向时）")
    parser.add_argument("--diagonal", "-d", choices=["left", "right"],
                        help="斜向水印: left=左上到右下, right=左下到右上")

    args = parser.parse_args()

    add_watermark(args.input, args.output, args.text, args.pos, args.diagonal)
