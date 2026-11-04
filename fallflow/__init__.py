"""Dense optical flow pipeline for sequences of frames (project "fall")."""

from .flow import FlowConfig, compute_flow, flow_stats, load_flow, save_flow
from .frames import discover_frames, load_frames, load_gray

__all__ = [
    "FlowConfig",
    "compute_flow",
    "flow_stats",
    "load_flow",
    "save_flow",
    "discover_frames",
    "load_frames",
    "load_gray",
]

__version__ = "0.1.0"
