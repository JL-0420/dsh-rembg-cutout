#!/usr/bin/env python3
"""u2net / u2netp 抠图脚本（供 DSH 插件 dsh-rembg-cutout 调用）。

不依赖 rembg 全家桶：只用 onnxruntime + numpy + Pillow，模型用 rembg 官方发布的
u2net / u2netp ONNX 权重（同样的预处理/后处理）。

用法：
  python cutout.py --input a.png --output a_cutout.png [--model u2netp] [--crop]
                   [--threshold 128] [--padding 0] [--feather 0] [--bg transparent|#ffffff]

输出：向 stdout 打一行 JSON（--json 亦同），字段见 result 构造处。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageFilter

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODELS = {
    # name -> (filename, download url)
    'u2netp': (
        'u2netp.onnx',
        'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx',
    ),
    'u2net': (
        'u2net.onnx',
        'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx',
    ),
}


def log(msg: str) -> None:
    print(f'[cutout] {msg}', file=sys.stderr, flush=True)


def ensure_model(name: str, model_dir: str) -> str:
    filename, url = MODELS[name]
    target = os.path.join(model_dir, filename)
    if os.path.isfile(target) and os.path.getsize(target) > 1024:
        return target
    os.makedirs(model_dir, exist_ok=True)
    log(f'模型 {name} 不存在，开始下载：{url}')
    tmp = target + '.part'
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, 'wb') as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    log(f'模型下载完成：{target} ({os.path.getsize(target)} bytes)')
    return target


def load_session(model_path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize((320, 320), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)


def predict(session, img: Image.Image) -> np.ndarray:
    ort_inputs = {session.get_inputs()[0].name: preprocess(img)}
    out = session.run(None, ort_inputs)[0]
    pred = out[0, 0, :, :]
    pred = pred - pred.min()
    span = pred.max() - pred.min()
    if span <= 1e-8:
        return np.zeros(img.size[::-1], dtype=np.uint8)
    pred = pred / span
    mask = Image.fromarray((pred * 255).astype(np.uint8), mode='L')
    mask = mask.resize(img.size, Image.LANCZOS)
    return np.asarray(mask, dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='u2netp', choices=sorted(MODELS))
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--threshold', type=int, default=128, help='alpha 阈值 0-255')
    ap.add_argument('--crop', action='store_true', help='裁到主体外框')
    ap.add_argument('--padding', type=int, default=0, help='裁剪时四周留白像素')
    ap.add_argument('--feather', type=float, default=0.0, help='alpha 羽化半径（像素）')
    ap.add_argument('--bg', default='transparent', help='transparent 或 #rrggbb')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    started = time.time()
    if not os.path.isfile(args.input):
        raise SystemExit(f'输入文件不存在：{args.input}')

    model_dir = args.model_dir or os.path.join(
        os.environ.get('DSH_HOME') or os.path.expanduser('~/.dsh'), 'rembg-cutout', 'models'
    )
    model_path = ensure_model(args.model, model_dir)

    session = load_session(model_path)
    src = Image.open(args.input)
    src.load()
    src = src.convert('RGBA')
    w, h = src.size

    alpha = predict(session, src)
    alpha = np.where(alpha >= max(1, args.threshold), 255, 0).astype(np.uint8)

    # 只保留最大的连通块-ish 处理：先看覆盖比例，过小时报错更友好
    coverage = float((alpha > 0).sum()) / float(w * h)

    out = src.copy()
    out.putalpha(Image.fromarray(alpha, mode='L'))
    if args.feather > 0:
        a_img = out.getchannel('A').filter(ImageFilter.GaussianBlur(radius=args.feather))
        out.putalpha(a_img)
        alpha = np.asarray(a_img, dtype=np.uint8)

    bbox = None
    if args.crop:
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0:
            raise SystemExit('没有抠出任何主体（alpha 全透明）。试试换模型（--model u2net）或降低 --threshold。')
        left = max(0, int(xs.min()) - args.padding)
        top = max(0, int(ys.min()) - args.padding)
        right = min(w, int(xs.max()) + 1 + args.padding)
        bottom = min(h, int(ys.max()) + 1 + args.padding)
        out = out.crop((left, top, right, bottom))
        bbox = {'left': left, 'top': top, 'width': right - left, 'height': bottom - top}

    if args.bg and args.bg.lower() != 'transparent':
        hexv = args.bg.lstrip('#')
        if len(hexv) != 6:
            raise SystemExit('--bg 需为 transparent 或 #rrggbb')
        color = tuple(int(hexv[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
        canvas = Image.new('RGBA', out.size, color)
        canvas.alpha_composite(out)
        out = canvas

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    fmt = 'PNG' if out.mode == 'RGBA' else None
    if args.output.lower().endswith(('.jpg', '.jpeg')):
        out.convert('RGB').save(args.output, quality=95)
    else:
        out.save(args.output, format=fmt or 'PNG')

    result = {
        'input': os.path.abspath(args.input),
        'output': os.path.abspath(args.output),
        'model': args.model,
        'inputSize': {'width': w, 'height': h},
        'outputSize': {'width': out.size[0], 'height': out.size[1]},
        'coverage': round(coverage, 4),
        'bbox': bbox,
        'elapsedMs': int((time.time() - started) * 1000),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
