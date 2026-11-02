"""
脱离 ComfyUI 的单元测试：验证打分与选择逻辑是否按预期工作。

跑法（任意装了 numpy 的 Python）：
    python test_core.py
ComfyUI 便携版建议用内置解释器跑：
    python_embeded\\python.exe test_core.py
"""

import numpy as np

from core import DEFAULT_WEIGHTS, format_report, score_images, select_indices


def _flat(h=64, w=64, value=0.5):
    return np.full((h, w, 3), value, dtype=np.float32)


def _gradient(h=64, w=64):
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    return np.repeat(np.repeat(ramp, h, axis=0), 3, axis=2)


def _checker(h=64, w=64, block=4, color=False):
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // block) + (xx // block)) % 2 == 0
    img = np.zeros((h, w, 3), dtype=np.float32)
    if color:
        img[mask] = (0.95, 0.15, 0.10)
        img[~mask] = (0.05, 0.35, 0.95)
    else:
        img[mask] = 0.9
        img[~mask] = 0.1
    return img


def main():
    names = ["纯色灰块", "平滑渐变", "黑白棋盘", "高饱和棋盘"]
    batch = np.stack([_flat(), _gradient(), _checker(), _checker(color=True)])

    scores, raw, norm = score_images(batch)
    kept = select_indices(scores, 2)
    report = format_report(scores, raw, kept)

    print(report)
    print()

    order = [names[i] for i in np.argsort(-scores)]
    print("排名：" + " > ".join(order))

    # 主排序：高饱和棋盘 > 黑白棋盘 > 渐变 > 纯色
    assert scores[3] > scores[2] > scores[1] > scores[0], "综合排序不符合直觉"
    assert set(kept.tolist()).issubset(set(range(4))), "索引越界"
    assert len(kept) == 2, "top_k 数量不对"
    assert list(kept) == sorted(kept), "输出索引应保持原始顺序"

    # 回归测试：红蓝两块亮度接近（0.38 vs 0.33），转灰度算锐度会误判成模糊。
    # 逐通道拉普拉斯必须能识别出色度边缘。
    red = np.zeros((64, 64, 3), dtype=np.float32); red[:] = (0.95, 0.15, 0.10)
    blue = np.zeros((64, 64, 3), dtype=np.float32); blue[:] = (0.05, 0.35, 0.95)
    chroma_edge = np.concatenate([np.tile(red, (1, 32, 1)), np.tile(blue, (1, 32, 1))], axis=1)
    _, raw_edge, _ = score_images(np.stack([chroma_edge]))
    gray_red = red[..., 0] * 0.299 + red[..., 1] * 0.587 + red[..., 2] * 0.114
    gray_blue = blue[..., 0] * 0.299 + blue[..., 1] * 0.587 + blue[..., 2] * 0.114
    lum_gap = abs(float(gray_red.mean()) - float(gray_blue.mean()))
    assert lum_gap < 0.08, f"构造的用例亮度差过大（{lum_gap:.3f}），无法复现该 bug"
    assert raw_edge["sharpness"][0] > 1.0, "色度边缘未被识别，逐通道锐度失效"

    mono, _, _ = score_images(np.stack([_checker(), _checker()]))
    assert np.allclose(mono, 0.5 * sum(DEFAULT_WEIGHTS.values())), "同分时归一化应退化为中性分"

    print("\n全部断言通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
