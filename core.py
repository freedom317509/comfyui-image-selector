"""
TraceLab Best-of-N：打分核心（纯 numpy 实现，不依赖 torch）

为什么要用 numpy 而不是 torch：
1. ComfyUI 环境一定自带 numpy，零额外依赖；
2. 核心打分逻辑可以脱离 ComfyUI 单独测试（见 test_core.py）；
3. 节点层只负责 torch.Tensor <-> numpy.ndarray 的转换，职责清晰。

图像约定：batch 形状 (B, H, W, C)，float32，取值范围 [0, 1]，与 ComfyUI 的 IMAGE 类型一致。
"""

from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {
    "sharpness": 0.50,
    "contrast": 0.25,
    "colorfulness": 0.25,
}


def _laplacian_variance(rgb: np.ndarray) -> np.ndarray:
    """
    逐通道 4 邻域拉普拉斯响应的方差，作为清晰度指标。数值越高 = 边缘越锐利。

    为什么是逐通道而不是先转灰度：v0.1 用 Rec.601 亮度图算，结果一张
    红蓝高饱和的图（两块亮度都接近 0.38 / 0.33）被判成"模糊"——亮度差小，
    色度边缘在灰度图里直接消失了。改成逐通道求方差再平均后修复。
    """
    per_channel = []
    for c in range(rgb.shape[-1]):
        g = rgb[..., c]
        center = 4.0 * g[:, 1:-1, 1:-1]
        lap = (
            center
            - g[:, :-2, 1:-1]
            - g[:, 2:, 1:-1]
            - g[:, 1:-1, :-2]
            - g[:, 1:-1, 2:]
        )
        per_channel.append(lap.reshape(lap.shape[0], -1).var(axis=1))
    return np.mean(per_channel, axis=0)


def _contrast(rgb: np.ndarray) -> np.ndarray:
    """逐通道标准差再平均，衡量画面的明暗与色彩层次。"""
    per_channel = [
        rgb[..., c].reshape(rgb.shape[0], -1).std(axis=1)
        for c in range(rgb.shape[-1])
    ]
    return np.mean(per_channel, axis=0)


def _colorfulness(batch: np.ndarray) -> np.ndarray:
    """
    Hasler & Süsstrunk (2003) 色彩丰富度：
    C = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)
    第一项是色彩变化量，第二项是平均色偏，比单纯统计饱和度更接近人眼感受。
    """
    rgb = batch[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg_std, yb_std = rg.std(axis=(1, 2)), yb.std(axis=(1, 2))
    rg_mean, yb_mean = rg.mean(axis=(1, 2)), yb.mean(axis=(1, 2))
    return np.sqrt(rg_std ** 2 + yb_std ** 2) + 0.3 * np.sqrt(rg_mean ** 2 + yb_mean ** 2)


def _minmax(x: np.ndarray) -> np.ndarray:
    """
    批内归一化。三个指标的原始量纲差异极大（清晰度动辄上千，对比度不到 1），
    不归一化的话加权求和会被清晰度完全支配。
    """
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def score_images(batch: np.ndarray, weights: dict | None = None):
    """
    返回 (总分数组, 原始指标字典, 归一化指标字典)。
    总分范围 [0, 1]，越大越好。
    """
    if batch.ndim != 4:
        raise ValueError(f"期望 (B,H,W,C) 的 4 维数组，实际收到 {batch.shape}")

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: v for k, v in weights.items() if v is not None})

    rgb = batch[..., :3]
    raw = {
        "sharpness": _laplacian_variance(rgb),
        "contrast": _contrast(rgb),
        "colorfulness": _colorfulness(batch),
    }
    norm = {k: _minmax(v) for k, v in raw.items()}

    total = np.zeros(batch.shape[0], dtype=np.float64)
    for key, values in norm.items():
        total += w.get(key, 0.0) * values
    return total, raw, norm


def select_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """
    取分数最高的 top_k 个下标，按原始顺序返回（保证输出顺序稳定、便于和文件名对应）。
    """
    n = len(scores)
    k = max(1, min(int(top_k), n))
    ranked = np.argsort(-scores, kind="stable")[:k]
    return np.sort(ranked)


def format_report(scores: np.ndarray, raw: dict, kept: np.ndarray) -> str:
    """生成可读的打分报告，直接输出到节点的 STRING 端口。"""
    kept_set = set(int(i) for i in kept)
    lines = [f"Best-of-{len(scores)} → 保留 {len(kept)} 张"]
    for i, s in enumerate(scores):
        mark = "✓" if i in kept_set else " "
        lines.append(
            f"{mark} #{i:<2d} 总分 {s:.3f} | 锐度 {raw['sharpness'][i]:8.2f} "
            f"| 对比 {raw['contrast'][i]:.3f} | 色彩 {raw['colorfulness'][i]:.2f}"
        )
    lines.append("保留索引: " + ", ".join(str(int(i)) for i in kept))
    return "\n".join(lines)
