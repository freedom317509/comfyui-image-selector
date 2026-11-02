"""
V3 节点定义（ComfyUI 新版 comfy_api.latest 接口）。

V3 的要点：
- 节点继承 io.ComfyNode，不再用 INPUT_TYPES / RETURN_TYPES / FUNCTION 这套类属性
- define_schema() 统一声明 node_id、显示名、分类、输入输出
- execute() 固定为 classmethod，返回 io.NodeOutput（而不是裸元组）
- 模块级导出 async def comfy_entrypoint()，由 ComfyUI 在扫描 custom_nodes 时调用
"""

from __future__ import annotations

import torch
from comfy_api.latest import ComfyExtension, io

from .core import DEFAULT_WEIGHTS, format_report, score_images, select_indices


class BestOfNSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TraceLab_BestOfNSelector",
            display_name="Best-of-N 图像优选 (TraceLab)",
            category="tracelab/select",
            description=(
                "对一批图像按锐度/对比度/色彩丰富度打分，只保留得分最高的 K 张。"
                "配合 batch 采样使用：一次生成 N 张，自动挑出最优的几张再送去放大或保存。"
            ),
            inputs=[
                io.Image.Input("images", tooltip="待筛选的图像批（batch）"),
                io.Int.Input(
                    "top_k", default=1, min=1, max=64, step=1,
                    tooltip="保留几张；超过批量大小时自动收敛到实际数量",
                ),
                io.Float.Input(
                    "w_sharpness", default=DEFAULT_WEIGHTS["sharpness"],
                    min=0.0, max=1.0, step=0.05, tooltip="锐度权重",
                ),
                io.Float.Input(
                    "w_contrast", default=DEFAULT_WEIGHTS["contrast"],
                    min=0.0, max=1.0, step=0.05, tooltip="对比度权重",
                ),
                io.Float.Input(
                    "w_colorfulness", default=DEFAULT_WEIGHTS["colorfulness"],
                    min=0.0, max=1.0, step=0.05, tooltip="色彩丰富度权重",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="selected"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        top_k: int,
        w_sharpness: float,
        w_contrast: float,
        w_colorfulness: float,
    ) -> io.NodeOutput:
        # ComfyUI 的 IMAGE 是 (B,H,W,C) 的 float tensor，转到 numpy 交给 core 打分
        batch = images.detach().cpu().float().numpy()
        weights = {
            "sharpness": w_sharpness,
            "contrast": w_contrast,
            "colorfulness": w_colorfulness,
        }
        scores, raw, _ = score_images(batch, weights)
        kept = select_indices(scores, top_k)

        idx = torch.as_tensor(kept, dtype=torch.long, device=images.device)
        return io.NodeOutput(images[idx], format_report(scores, raw, kept))


class TraceLabExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BestOfNSelector]


async def comfy_entrypoint() -> ComfyExtension:
    return TraceLabExtension()
