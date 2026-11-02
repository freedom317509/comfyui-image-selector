"""
V1 节点定义（旧版 NODE_CLASS_MAPPINGS 接口），用于兼容较老的 ComfyUI。

如果你的 ComfyUI 版本较新，node 会自动走 nodes_v3.py，这个文件不会被加载。
只有当启动日志提示 V3 不可用时，__init__.py 才会回退到这里。
"""

from __future__ import annotations

import torch

from .core import DEFAULT_WEIGHTS, format_report, score_images, select_indices


class BestOfNSelectorV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "w_sharpness": ("FLOAT", {
                    "default": DEFAULT_WEIGHTS["sharpness"], "min": 0.0, "max": 1.0,
                    "step": 0.05, "display": "slider",
                }),
                "w_contrast": ("FLOAT", {
                    "default": DEFAULT_WEIGHTS["contrast"], "min": 0.0, "max": 1.0,
                    "step": 0.05, "display": "slider",
                }),
                "w_colorfulness": ("FLOAT", {
                    "default": DEFAULT_WEIGHTS["colorfulness"], "min": 0.0, "max": 1.0,
                    "step": 0.05, "display": "slider",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected", "report")
    FUNCTION = "run"
    CATEGORY = "tracelab/select"

    def run(self, images, top_k, w_sharpness, w_contrast, w_colorfulness):
        batch = images.detach().cpu().float().numpy()
        weights = {
            "sharpness": w_sharpness,
            "contrast": w_contrast,
            "colorfulness": w_colorfulness,
        }
        scores, raw, _ = score_images(batch, weights)
        kept = select_indices(scores, top_k)

        idx = torch.as_tensor(kept, dtype=torch.long, device=images.device)
        return (images[idx], format_report(scores, raw, kept))


NODE_CLASS_MAPPINGS = {"TraceLab_BestOfNSelector": BestOfNSelectorV1}
NODE_DISPLAY_NAME_MAPPINGS = {"TraceLab_BestOfNSelector": "Best-of-N 图像优选 (TraceLab)"}
