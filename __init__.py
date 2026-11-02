"""
TraceLab Best-of-N 图像优选 —— ComfyUI 自定义节点入口。

ComfyUI 扫描 custom_nodes 时会 import 本包：
- 新版（含 comfy_api）走 V3：本模块导出 comfy_entrypoint，ComfyUI 调它取节点列表
- 老版没有 comfy_api，import 失败 → 回退到 V1 的 NODE_CLASS_MAPPINGS
"""

V3_LOADED = False
_V3_ERROR: Exception | None = None

try:
    from .nodes_v3 import comfy_entrypoint  # noqa: F401  ComfyUI 从这里发现节点

    V3_LOADED = True
except Exception as exc:  # 老版本 ComfyUI 没有 comfy_api.latest
    _V3_ERROR = exc

if not V3_LOADED:
    from .nodes_v1 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401

    print(f"[TraceLab] V3 接口不可用（{_V3_ERROR}），已回退到 V1 注册方式。")

__all__ = ["V3_LOADED"]
