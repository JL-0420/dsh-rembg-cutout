#!/usr/bin/env python3
"""自动裁掉图片四周的纯色留白（或按透明通道裁到内容）。

用法：
  python trim.py --input a.png --output a_trim.png [--tolerance 8] [--mode auto|alpha|color]
                 [--padding 0]

输出：stdout 一行 JSON。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image


def dominant_border_color(rgba: np.ndarray) -> np.ndarray:
    """取四条边上像素的众数颜色，当作背景色。"""
    h, w, _ = rgba.shape
    border = np.concatenate([
        rgba[0, :, :3].reshape(-1, 3),
        rgba[h - 1, :, :3].reshape(-1, 3),
        rgba[:, 0, :3].reshape(-1, 3),
        rgba[:, w - 1, :3].reshape(-1, 3),
    ]).astype(np.int16)
    if border.size == 0:
        return np.array([255, 255, 255], dtype=np.int16)
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return colors[int(np.argmax(counts))]


def compute_bbox(rgba: np.ndarray, mode: str, tolerance: int):
    h, w, _ = rgba.shape
    alpha = rgba[:, :, 3].astype(np.int16)

    if mode in ('auto', 'alpha') and alpha.min() < 250:
        content = alpha > 0
    elif mode == 'alpha':
        return None, 'alpha 通道全不透明，没有可裁的透明区域'
    else:
        bg = dominant_border_color(rgba)
        diff = np.abs(rgba[:, :, :3].astype(np.int16) - bg[None, None, :]).max(axis=2)
        content = (diff > max(0, int(tolerance))) | (alpha <= 0)

    rows = content.any(axis=1)
    cols = content.any(axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    if len(xs) == 0 or len(ys) == 0:
        return None, '整张图都是背景色，没有可裁的内容'
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--tolerance', type=int, default=8)
    ap.add_argument('--mode', default='auto', choices=['auto', 'alpha', 'color'])
    ap.add_argument('--padding', type=int, default=0)
    ap.add_argument('--json', action='store_true', help='保留参数，输出始终为一行 JSON')
    args = ap.parse_args()

    started = time.time()
    if not os.path.isfile(args.input):
        raise SystemExit(f'输入文件不存在：{args.input}')

    img = Image.open(args.input)
    img.load()
    rgba_src = img.convert('RGBA')
    arr = np.asarray(rgba_src)
    h, w = arr.shape[:2]

    bbox, error = compute_bbox(arr, args.mode, args.tolerance)
    if bbox is None:
        raise SystemExit(error)

    left, top, right, bottom = bbox
    pad = max(0, int(args.padding))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(w, right + pad)
    bottom = min(h, bottom + pad)
    out = rgba_src.crop((left, top, right, bottom))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    if args.output.lower().endswith(('.jpg', '.jpeg')):
        out.convert('RGB').save(args.output, quality=95)
    else:
        out.save(args.output, format='PNG')

    print(json.dumps({
        'input': os.path.abspath(args.input),
        'output': os.path.abspath(args.output),
        'mode': args.mode,
        'inputSize': {'width': w, 'height': h},
        'outputSize': {'width': out.size[0], 'height': out.size[1]},
        'bbox': {'left': left, 'top': top, 'width': right - left, 'height': bottom - top},
        'elapsedMs': int((time.time() - started) * 1000),
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
