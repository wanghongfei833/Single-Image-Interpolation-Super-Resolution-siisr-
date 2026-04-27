"""
SII-SLS 图像超分辨率工具 v2x
基于 Semi-local Similarity 的单图像插值算法
"""

import sys
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QStyleFactory
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap

from siisls import interpolate_2x_array, Params2x
from siisls.utils import psnr as calc_psnr, ssim as calc_ssim, to_gray_float


class ProcessingThread(QThread):
    """处理线程 - 避免界面卡顿"""
    progress = Signal(float)
    status = Signal(str)
    finished = Signal(bool, str)
    preview_ready = Signal(np.ndarray)
    metrics_ready = Signal(float, float, bool)  # PSNR, SSIM, has_reference
    error_dialog = Signal(str, str)  # title, message - 用于显示错误对话框

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_cancelled = False
        self.task_type = None
        self.input_path = None
        self.gt_path = None  # 对照图像路径
        self.output_path = None
        self.params = None
        self.color_mode = "rgb"

    def cancel(self):
        self._is_cancelled = True

    def setup(self, task_type, input_path, gt_path, output_path, params, color_mode="rgb"):
        self.task_type = task_type
        self.input_path = input_path
        self.gt_path = gt_path
        self.output_path = output_path
        self.params = params
        self.color_mode = color_mode

    def _make_progress_callback(self, base_progress: float, range_size: float):
        """创建分阶段的进度回调"""
        def callback(progress: float, msg: str):
            if self._is_cancelled:
                return
            overall = base_progress + progress * range_size
            self.progress.emit(overall)
            self.status.emit(msg)
        return callback

    def run(self):
        try:
            if self.task_type == 'image':
                self._process_image()
        except Exception as e:
            self.finished.emit(False, f"处理失败: {str(e)}\n{traceback.format_exc()}")

    def _load_image(self, path):
        """加载图像并标准化"""
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def _process_image(self):
        self.status.emit("正在读取图像...")
        self.progress.emit(0.1)

        self.status.emit(f"[DEBUG] 输入路径: {self.input_path}")
        self.status.emit(f"[DEBUG] GT路径: {self.gt_path}")
        img_array = self._load_image(self.input_path)
        h, w = img_array.shape[:2]
        self.status.emit(f"[DEBUG] LR原始尺寸: {w}x{h}, shape={img_array.shape}")
        self.status.emit(f"实验图像(LR)尺寸: {w}x{h}")

        # 检查最小尺寸要求
        min_size = 64
        if h < min_size or w < min_size:
            self.finished.emit(False, f"图像尺寸太小 ({w}x{h})，至少需要 {min_size}x{min_size}")
            return

        # 检查是否为4的倍数（算法内部需要）
        if h % 4 != 0 or w % 4 != 0:
            self.error_dialog.emit(
                "尺寸不符合要求",
                f"实验图像尺寸必须是4的倍数。\n\n"
                f"当前尺寸: {w}x{h}\n"
                f"提示：图像尺寸必须是4的倍数（宽高都能被4整除）\n"
                f"请使用\"制作数据\"按钮重新生成符合要求的图像。"
            )
            self.finished.emit(False, "图像尺寸不符合要求（需为4的倍数）")
            return

        # 加载对照图像（Ground Truth）
        gt_array = None
        has_reference = False
        if self.gt_path and os.path.exists(self.gt_path):
            try:
                gt_array = self._load_image(self.gt_path)
                gt_h, gt_w = gt_array.shape[:2]
                self.status.emit(f"[DEBUG] GT原始尺寸: {gt_w}x{gt_h}, shape={gt_array.shape}")

                # 验证尺寸：对照图像应该是实验图像的2倍
                expected_h, expected_w = h * 2, w * 2
                if gt_h != expected_h or gt_w != expected_w:
                    self.finished.emit(
                        False,
                        f"尺寸不匹配！\n\n"
                        f"实验图像(LR): {w}x{h}\n"
                        f"对照图像(GT): {gt_w}x{gt_h}\n"
                        f"期望GT: {expected_w}x{expected_h} (应为LR的2倍)\n\n"
                        f"提示：使用\"制作数据\"按钮生成匹配的LR和GT图像对。"
                    )
                    return

                has_reference = True
                self.status.emit("已加载对照图像，尺寸验证通过")
                self.status.emit(f"[DEBUG] 尺寸匹配: LR({w}x{h}) + GT({gt_w}x{gt_h})")
            except Exception as e:
                self.status.emit(f"加载对照图像失败: {e}")

        self.status.emit("正在进行2x超分辨率处理...")
        self.progress.emit(1.0)

        # 使用带回调的interpolate_2x_array
        result = interpolate_2x_array(
            img_array,
            params=self.params,
            return_intermediates=False,
            color_mode=self.color_mode,
            progress_callback=self._make_progress_callback(0.01, 0.90),
        )

        self.progress.emit(92.0)
        result_h, result_w = result.shape[:2]
        self.status.emit(f"[DEBUG] 超分结果尺寸: {result_w}x{result_h}, shape={result.shape}")

        # 计算精度
        if has_reference and gt_array is not None:
            # 对照模式：对比超分结果和对照图像
            gt_gray = to_gray_float(gt_array)
            result_gray = to_gray_float(result) if result.ndim == 3 else result
            self.status.emit(f"[DEBUG] gt_gray shape: {gt_gray.shape}, result_gray shape: {result_gray.shape}")
            self.status.emit(f"[DEBUG] gt_gray元素数: {gt_gray.size}, result_gray元素数: {result_gray.size}")
            p = calc_psnr(result_gray, gt_gray)
            s = calc_ssim(result_gray, gt_gray)
            self.metrics_ready.emit(p, s, True)
            self.status.emit(f"处理完成 PSNR={p:.2f}dB SSIM={s:.4f}")
        else:
            # 无对照模式
            self.metrics_ready.emit(0.0, 0.0, False)
            self.status.emit("处理完成（无对照图像，无法计算PSNR/SSIM）")

        self.progress.emit(95.0)
        self.preview_ready.emit(np.clip(result, 0, 255).astype(np.uint8))

        # 保存结果
        self.status.emit("正在保存结果...")
        result_uint8 = np.clip(result, 0, 255).round().astype(np.uint8)

        out_h, out_w = result_uint8.shape[:2]
        self.status.emit(f"输出图像尺寸: {out_w}x{out_h}")

        if result_uint8.ndim == 3 and result_uint8.shape[2] == 3:
            Image.fromarray(result_uint8, mode='RGB').save(self.output_path)
        else:
            gray = result_uint8.squeeze() if result_uint8.ndim == 3 else result_uint8
            Image.fromarray(gray).save(self.output_path)

        self.progress.emit(100.0)

        if has_reference:
            self.finished.emit(True, f"处理完成!\n输出: {self.output_path}\nPSNR: {p:.2f} dB\nSSIM: {s:.4f}")
        else:
            self.finished.emit(True, f"处理完成!\n输出: {self.output_path}\n注意: 未提供对照图像，无法计算PSNR/SSIM")

    def _process_video(self):
        self.status.emit("正在打开视频...")
        self.progress.emit(0.1)

        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            self.finished.emit(False, "无法打开视频文件")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.status.emit(f"视频: {width}x{height}, {total_frames}帧, {fps:.1f}fps")
        self.progress.emit(0.5)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width*2, height*2))

        frame_idx = 0
        psnr_sum = 0.0
        ssim_sum = 0.0
        psnr_min = float('inf')
        psnr_max = float('-inf')

        while not self._is_cancelled:
            ret, frame = cap.read()
            if not ret:
                break

            # 每10帧报告一次进度
            frame_progress = frame_idx / total_frames
            self.progress.emit(frame_progress * 0.95 + 0.5)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = to_gray_float(frame_rgb)

            # 处理当前帧
            result = interpolate_2x_array(
                frame_rgb,
                params=self.params,
                color_mode=self.color_mode,
            )
            result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(result_bgr)

            # 计算当前帧精度
            result_gray = to_gray_float(result) if result.ndim == 3 else result
            p = calc_psnr(result_gray, frame_gray)
            s = calc_ssim(result_gray, frame_gray)

            psnr_sum += p
            ssim_sum += s
            psnr_min = min(psnr_min, p)
            psnr_max = max(psnr_max, p)
            frame_idx += 1

            # 每5帧更新一次UI
            if frame_idx % 5 == 0:
                avg_psnr = psnr_sum / frame_idx
                avg_ssim = ssim_sum / frame_idx
                self.metrics_ready.emit(avg_psnr, avg_ssim)
                self.status.emit(
                    f"帧{frame_idx}/{total_frames} "
                    f"PSNR={p:.1f}dB(均{avg_psnr:.1f}) SSIM={s:.4f}"
                )
                self.preview_ready.emit(np.clip(result, 0, 255).astype(np.uint8))
                self.msleep(1)

        self._is_cancelled = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.gt_path = None      # 对照图像（高分辨率）
        self.input_path = None   # 实验图像（低分辨率）
        self.processing_thread = None
        self.current_psnr = 0.0
        self.current_ssim = 0.0
        self.output_dir = str(Path.home() / "SII_SLS_Output")
        self.setup_ui()
        self.setup_thread()

    def setup_ui(self):
        self.setWindowTitle("SII-SLS 超分辨率工具 v2x")
        self.setMinimumSize(900, 800)
        self.setStyle(QStyleFactory.create("Fusion"))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ===== 图像选择区 =====
        file_group = QGroupBox("图像选择")
        file_layout = QVBoxLayout()

        # 对照图像选择
        gt_layout = QHBoxLayout()
        self.gt_label = QLabel("未选择对照图像（高分辨率）")
        self.gt_label.setStyleSheet("QLabel { background: #f0f0f0; padding: 8px; border-radius: 3px; }")
        self.gt_label.setMinimumHeight(35)
        self.gt_label.setWordWrap(True)
        btn_gt = QPushButton("对照图像 (GT)")
        btn_gt.setMinimumHeight(35)
        btn_gt.clicked.connect(self.select_gt)
        btn_gt_clear = QPushButton("✕")
        btn_gt_clear.setMaximumWidth(35)
        btn_gt_clear.clicked.connect(self.clear_gt)
        gt_layout.addWidget(QLabel("对照图像:"))
        gt_layout.addWidget(self.gt_label, 1)
        gt_layout.addWidget(btn_gt)
        gt_layout.addWidget(btn_gt_clear)
        file_layout.addLayout(gt_layout)

        # 实验图像选择
        input_layout = QHBoxLayout()
        self.input_label = QLabel("未选择实验图像（低分辨率）")
        self.input_label.setStyleSheet("QLabel { background: #e8f5e9; padding: 8px; border-radius: 3px; }")
        self.input_label.setMinimumHeight(35)
        self.input_label.setWordWrap(True)
        btn_input = QPushButton("实验图像 (LR)")
        btn_input.setMinimumHeight(35)
        btn_input.clicked.connect(self.select_input)
        btn_input_clear = QPushButton("✕")
        btn_input_clear.setMaximumWidth(35)
        btn_input_clear.clicked.connect(self.clear_input)
        input_layout.addWidget(QLabel("实验图像:"))
        input_layout.addWidget(self.input_label, 1)
        input_layout.addWidget(btn_input)
        input_layout.addWidget(btn_input_clear)
        file_layout.addLayout(input_layout)

        # 输出目录
        out_layout = QHBoxLayout()
        self.out_label = QLabel(self.output_dir)
        self.out_label.setStyleSheet("QLabel { background: #f0f0f0; padding: 5px; border-radius: 3px; }")
        self.out_label.setMinimumHeight(30)
        btn_out = QPushButton("输出目录")
        btn_out.clicked.connect(self.select_output)
        out_layout.addWidget(QLabel("输出:"))
        out_layout.addWidget(self.out_label, 1)
        out_layout.addWidget(btn_out)
        file_layout.addLayout(out_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # ===== 参数设置区 =====
        param_group = QGroupBox("参数设置")
        param_layout = QHBoxLayout()

        left = QVBoxLayout()
        self.spin_n1 = QSpinBox(); self.spin_n1.setRange(4, 16); self.spin_n1.setValue(8)
        self.spin_n2 = QSpinBox(); self.spin_n2.setRange(4, 16); self.spin_n2.setValue(6)
        self.spin_K = QSpinBox(); self.spin_K.setRange(1, 50); self.spin_K.setValue(12)
        left.addWidget(QLabel("n1 (补丁大小)")); left.addWidget(self.spin_n1)
        left.addWidget(QLabel("n2 (补丁大小)")); left.addWidget(self.spin_n2)
        left.addWidget(QLabel("K (近邻数)")); left.addWidget(self.spin_K)

        mid = QVBoxLayout()
        self.spin_iter1 = QSpinBox(); self.spin_iter1.setRange(1, 10); self.spin_iter1.setValue(2)
        self.spin_iter2 = QSpinBox(); self.spin_iter2.setRange(1, 10); self.spin_iter2.setValue(2)
        self.spin_iter3 = QSpinBox(); self.spin_iter3.setRange(1, 20); self.spin_iter3.setValue(6)
        mid.addWidget(QLabel("迭代次数1")); mid.addWidget(self.spin_iter1)
        mid.addWidget(QLabel("迭代次数2")); mid.addWidget(self.spin_iter2)
        mid.addWidget(QLabel("迭代次数3")); mid.addWidget(self.spin_iter3)

        right = QVBoxLayout()
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 5.0); self.spin_sigma.setValue(0.85); self.spin_sigma.setSingleStep(0.05)
        self.spin_cw = QSpinBox()
        self.spin_cw.setRange(100, 2000); self.spin_cw.setValue(500)
        self.chk_preview = QCheckBox("实时预览"); self.chk_preview.setChecked(True)
        self.chk_color = QCheckBox("彩色处理"); self.chk_color.setChecked(True)
        right.addWidget(QLabel("sigma (平滑)")); right.addWidget(self.spin_sigma)
        right.addWidget(QLabel("CW (权重)")); right.addWidget(self.spin_cw)
        right.addWidget(self.chk_preview)
        right.addWidget(self.chk_color)

        param_layout.addLayout(left)
        param_layout.addLayout(mid)
        param_layout.addLayout(right)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # ===== 预览区 =====
        preview_group = QGroupBox("处理预览")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("请先选择实验图像...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(250)
        self.preview_label.setStyleSheet("""
            QLabel { background: #2a2a2a; color: #ffffff; border-radius: 5px; padding: 10px; font-size: 14px; }
        """)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # ===== 精度显示区 =====
        metrics_group = QGroupBox("处理精度")
        metrics_layout = QHBoxLayout()

        self.psnr_label = QLabel("PSNR: -- dB")
        self.psnr_label.setAlignment(Qt.AlignCenter)
        self.psnr_label.setStyleSheet("""
            QLabel {
                background: #1a1a2e;
                color: #00ff88;
                font-size: 18px;
                font-weight: bold;
                font-family: monospace;
                padding: 8px;
                border-radius: 5px;
            }
        """)

        self.ssim_label = QLabel("SSIM: --")
        self.ssim_label.setAlignment(Qt.AlignCenter)
        self.ssim_label.setStyleSheet("""
            QLabel {
                background: #1a1a2e;
                color: #ffaa00;
                font-size: 18px;
                font-weight: bold;
                font-family: monospace;
                padding: 8px;
                border-radius: 5px;
            }
        """)

        metrics_layout.addWidget(self.psnr_label, 1)
        metrics_layout.addWidget(self.ssim_label, 1)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # ===== 进度条 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setFormat("%p% - %s")
        layout.addWidget(self.progress_bar)

        # ===== 控制按钮 =====
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("🚀 开始超分辨率")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.setStyleSheet("""
            QPushButton { background: #4CAF50; color: white; font-size: 15px; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background: #45a049; }
            QPushButton:disabled { background: #cccccc; }
        """)
        self.btn_start.clicked.connect(self.start_processing)

        self.btn_cancel = QPushButton("⏹️ 取消")
        self.btn_cancel.setMinimumHeight(45)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("""
            QPushButton { background: #f44336; color: white; font-size: 15px; font-weight: bold; border-radius: 8px; }
            QPushButton:disabled { background: #cccccc; }
        """)
        self.btn_cancel.clicked.connect(self.cancel_processing)

        self.btn_mkdata = QPushButton("📊 制作数据")
        self.btn_mkdata.setMinimumHeight(45)
        self.btn_mkdata.setStyleSheet("""
            QPushButton { background: #2196F3; color: white; font-size: 15px; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background: #1976D2; }
        """)
        self.btn_mkdata.clicked.connect(self.make_dataset)

        btn_layout.addWidget(self.btn_mkdata)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        # ===== 日志 =====
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(80)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { background: #1e1e1e; color: #00ff00; font-family: monospace; }")
        layout.addWidget(self.log_text)

        self.log("系统就绪，请选择实验图像...")

    def setup_thread(self):
        self.processing_thread = ProcessingThread(self)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.status.connect(self.update_status)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.preview_ready.connect(self.update_preview)
        self.processing_thread.metrics_ready.connect(self.update_metrics)
        self.processing_thread.error_dialog.connect(self.show_error_dialog)

    def show_error_dialog(self, title, message):
        """显示错误对话框（由处理线程调用）"""
        QMessageBox.warning(self, title, message, QMessageBox.Ok)

    def log(self, msg):
        self.log_text.append(f"[{self._time()}] {msg}")

    def _time(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def select_gt(self):
        """选择对照图像（高分辨率）"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择对照图像（高分辨率）", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.gt_path = path
            self.gt_label.setText(f"✓ {Path(path).name}")
            self.gt_label.setStyleSheet("QLabel { background: #c8e6c9; padding: 8px; border-radius: 3px; }")
            self.log(f"已选择对照图像: {path}")
            self._update_start_button_state()

    def clear_gt(self):
        """清除对照图像"""
        self.gt_path = None
        self.gt_label.setText("未选择对照图像（高分辨率）")
        self.gt_label.setStyleSheet("QLabel { background: #f0f0f0; padding: 8px; border-radius: 3px; }")
        self.log("已清除对照图像")
        self._update_start_button_state()

    def select_input(self):
        """选择实验图像（低分辨率）"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择实验图像（低分辨率）", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.input_path = path
            self.input_label.setText(f"✓ {Path(path).name}")
            self.input_label.setStyleSheet("QLabel { background: #a5d6a7; padding: 8px; border-radius: 3px; }")
            self.log(f"已选择实验图像: {path}")
            self._show_input_preview(path)
            self._update_start_button_state()

    def clear_input(self):
        """清除实验图像"""
        self.input_path = None
        self.input_label.setText("未选择实验图像（低分辨率）")
        self.input_label.setStyleSheet("QLabel { background: #e8f5e9; padding: 8px; border-radius: 3px; }")
        self.preview_label.setText("请先选择实验图像...")
        self.preview_label.setPixmap(QPixmap())
        self._reset_metrics()
        self._update_start_button_state()

    def _update_start_button_state(self):
        """更新开始按钮状态"""
        self.btn_start.setEnabled(self.input_path is not None)

    def _show_input_preview(self, path):
        """显示实验图像预览"""
        try:
            img = Image.open(path)
            frame = np.array(img)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            h, w = frame.shape[:2]
            ch = 3 if frame.ndim == 3 else 1

            max_size = 400
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                h, w = frame.shape[:2]

            if ch == 3:
                qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            else:
                qt_img = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)

            self.preview_label.setPixmap(QPixmap.fromImage(qt_img))
            self.preview_label.setText("")
            self.log(f"实验图像尺寸: {img.size[0]}x{img.size[1]}")
        except Exception as e:
            self.log(f"预览失败: {e}")

    def _reset_metrics(self):
        """重置精度显示"""
        self.psnr_label.setText("PSNR: -- dB")
        self.ssim_label.setText("SSIM: --")
        self.psnr_label.setStyleSheet("""
            QLabel {
                background: #1a1a2e;
                color: #00ff88;
                font-size: 18px;
                font-weight: bold;
                font-family: monospace;
                padding: 8px;
                border-radius: 5px;
            }
        """)
        self.current_psnr = 0.0
        self.current_ssim = 0.0

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", "")
        if path:
            self.output_dir = path
            self.out_label.setText(path)
            self.log(f"输出目录: {path}")

    def get_params(self):
        return Params2x(
            n1=self.spin_n1.value(),
            n2=self.spin_n2.value(),
            K=self.spin_K.value(),
            sigma=self.spin_sigma.value(),
            cw=float(self.spin_cw.value()),
            iter1=self.spin_iter1.value(),
            iter2=self.spin_iter2.value(),
            iter3=self.spin_iter3.value(),
            W1=20, W2=20, W3=30, W4=30
        )

    def make_dataset(self):
        """制作数据集：将高分辨率图像下采样生成低分辨率图像"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择高分辨率图像（将生成对应的低分辨率版本）", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        try:
            self.log("正在加载图像...")
            img = Image.open(path)
            img_array = np.array(img)

            h, w = img_array.shape[:2]
            if img_array.ndim == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]

            self.log(f"原图尺寸: {w}x{h}")

            # 计算满足条件的尺寸:
            # 1. GT必须是8的倍数（因为GT=LR*2, LR需是4的倍数）
            # 2. 向下取整到8的倍数
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8

            # 如果尺寸变化过大，说明原图太小
            min_size = 64
            if new_h < min_size or new_w < min_size:
                QMessageBox.warning(
                    self, "图像太小",
                    f"图像尺寸 {w}x{h} 太小，制作数据失败。\n"
                    f"需要至少 {min_size}x{min_size} 的图像（且宽高需能被8整除）",
                    QMessageBox.Ok
                )
                return

            # 裁剪原图到修正尺寸
            if new_h != h or new_w != w:
                img_array = img_array[:new_h, :new_w]
                self.log(f"修正尺寸: {new_w}x{new_h}")

            # 保存修正后的GT图像 (新尺寸是8的倍数)
            input_path = Path(path)
            gt_path = input_path.parent / f"{input_path.stem}_gt{input_path.suffix}"
            gt_uint8 = np.clip(img_array, 0, 255).round().astype(np.uint8)
            if gt_uint8.ndim == 3 and gt_uint8.shape[2] == 3:
                Image.fromarray(gt_uint8, mode='RGB').save(gt_path)
            else:
                Image.fromarray(gt_uint8.squeeze() if gt_uint8.ndim == 3 else gt_uint8).save(gt_path)

            # 下采样生成LR图像
            lr_array = img_array[0::2, 0::2]

            # 保存LR图像
            lr_path = input_path.parent / f"{input_path.stem}_lr{input_path.suffix}"
            lr_uint8 = np.clip(lr_array, 0, 255).round().astype(np.uint8)
            if lr_uint8.ndim == 3 and lr_uint8.shape[2] == 3:
                Image.fromarray(lr_uint8, mode='RGB').save(lr_path)
            else:
                Image.fromarray(lr_uint8.squeeze() if lr_uint8.ndim == 3 else lr_uint8).save(lr_path)

            gt_h, gt_w = img_array.shape[:2]
            lr_h, lr_w = lr_array.shape[:2]
            self.log(f"已生成数据对:")
            self.log(f"  GT: {gt_path} ({gt_w}x{gt_h})")
            self.log(f"  LR: {lr_path} ({lr_w}x{lr_h})")

            QMessageBox.information(
                self, "制作完成",
                f"已生成数据对\n\n"
                f"GT (高分辨率): {gt_w}x{gt_h}\n"
                f"LR (低分辨率): {lr_w}x{lr_h}\n\n"
                f"文件:\n{gt_path}\n{lr_path}"
            )

        except Exception as e:
            self.log(f"制作数据失败: {e}")
            QMessageBox.warning(self, "制作失败", f"处理失败: {str(e)}")

    def start_processing(self):
        if not self.input_path:
            QMessageBox.warning(self, "提示", "请先选择实验图像！")
            return

        self._reset_metrics()
        os.makedirs(self.output_dir, exist_ok=True)

        input_name = Path(self.input_path).stem
        output_path = os.path.join(self.output_dir, f"{input_name}_sr.png")

        color_mode = "rgb" if self.chk_color.isChecked() else "gray"
        color_info = "彩色" if self.chk_color.isChecked() else "灰度"

        gt_info = "已提供" if self.gt_path else "未提供"
        if not self.gt_path:
            QMessageBox.information(
                self, "注意",
                "未提供对照图像，将无法计算PSNR/SSIM指标！\n\n如需计算指标，请在\"对照图像\"中选择对应的 Ground Truth 图像。",
                QMessageBox.Ok
            )

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("确认处理")
        msg.setText(f"实验图像: {self.input_path}\n对照图像: {gt_info}\n\n输出至:\n{output_path}")
        msg.setInformativeText(f"处理模式: 2x超分辨率 ({color_info})")
        btn_ok = msg.addButton("开始处理", QMessageBox.AcceptRole)
        btn_cancel = msg.addButton("取消", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_ok)
        msg.exec()

        if msg.clickedButton() != btn_ok:
            self.log("用户取消")
            return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.processing_thread.setup(
            'image',
            self.input_path,
            self.gt_path,
            output_path,
            self.get_params(),
            color_mode
        )
        self.processing_thread.start()
        self.log("开始超分辨率处理...")

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认取消",
                "确定要取消当前处理吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processing_thread.cancel()
                self.log("正在取消...")
                self.btn_cancel.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(int(value * 10))

    def update_status(self, status):
        self.progress_bar.setFormat(f"%p% - {status}")

    def update_metrics(self, psnr, ssim, has_reference):
        """更新精度显示"""
        self.current_psnr = psnr
        self.current_ssim = ssim
        if has_reference:
            self.psnr_label.setText(f"PSNR: {psnr:.2f} dB")
            self.ssim_label.setText(f"SSIM: {ssim:.4f}")
        else:
            self.psnr_label.setText("PSNR: --")
            self.ssim_label.setText("SSIM: --")
            self.psnr_label.setStyleSheet("""
                QLabel {
                    background: #4a4a4a;
                    color: #888888;
                    font-size: 18px;
                    font-weight: bold;
                    font-family: monospace;
                    padding: 8px;
                    border-radius: 5px;
                }
            """)
            self.ssim_label.setStyleSheet("""
                QLabel {
                    background: #4a4a4a;
                    color: #888888;
                    font-size: 18px;
                    font-weight: bold;
                    font-family: monospace;
                    padding: 8px;
                    border-radius: 5px;
                }
            """)

    def update_preview(self, frame):
        if not self.chk_preview.isChecked():
            return
        try:
            h, w = frame.shape[:2]
            self.log(f"超分结果尺寸: {w}x{h}")

            max_size = 400
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                display_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            else:
                display_frame = frame.copy()

            if display_frame.ndim == 3 and display_frame.shape[2] == 3:
                dh, dw, ch = display_frame.shape
                qt_img = QImage(display_frame.data, dw, dh, ch * dw, QImage.Format_RGB888)
            else:
                dh, dw = display_frame.shape[:2]
                qt_img = QImage(display_frame.data, dw, dh, dw, QImage.Format_Grayscale8)

            self.preview_label.setPixmap(QPixmap.fromImage(qt_img))
            self.preview_label.setText("")
        except Exception as e:
            self.log(f"预览失败: {e}")

    def processing_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setValue(1000 if success else self.progress_bar.value())
        self.log(msg)

        if success:
            self.psnr_label.setStyleSheet("""
                QLabel {
                    background: #004d00;
                    color: #00ff88;
                    font-size: 18px;
                    font-weight: bold;
                    font-family: monospace;
                    padding: 8px;
                    border-radius: 5px;
                    border: 2px solid #00ff88;
                }
            """)
            QMessageBox.information(self, "处理完成", msg)
        else:
            QMessageBox.warning(self, "处理失败", msg)

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, "退出确认",
                "正在处理中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processing_thread.cancel()
                self.processing_thread.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SII-SLS 超分辨率工具")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
