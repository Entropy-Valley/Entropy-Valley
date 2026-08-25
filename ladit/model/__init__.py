"""LaDiT model module.

Re-exports the unified backbone loader so callers can do
`from ladit.model import load_backbone`.
"""

from .backbone import load_backbone, build_attention_mask

__all__ = ["load_backbone", "build_attention_mask"]
