# OmnimatteZero (comfy-native)
# Native LTXV implementation: removal / total-mask / foreground-compose,
# running on comfy's own LTXV model loading, VAE and memory management.
# The legacy diffusers-wrapper implementation is kept in the plugin root
# files (unused) for reference.

from .oz_native.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
