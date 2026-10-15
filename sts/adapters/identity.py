"""Identity slots (R1): `identity_resolver` and `identity_gallery`.

Rule 2 of docs/ARCHITECTURE.md: this adapter only constructs objects from poi_localization.identity;
it contains no identity maths.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional


class NoIdentity:
    """identity_resolver=none: M4 runs exactly as before (no engine)."""

    def build(self, m4_cfg, gallery=None, log_dir: Optional[Path] = None, cam_id: str = "cam0"):
        return None


class MulticueV1:
    """identity_resolver=multicue_v1: builds the IdentityEngine iff M4Config.identity.enabled."""

    def build(self, m4_cfg, gallery=None, log_dir: Optional[Path] = None, cam_id: str = "cam0"):
        icfg = m4_cfg.identity
        if not icfg.enabled:
            return None
        from poi_localization.identity.engine import IdentityEngine
        from poi_localization.identity.io import IdentityJsonlWriter
        writer = None
        if log_dir is not None and icfg.log.enabled:
            writer = IdentityJsonlWriter(Path(log_dir) / f"{cam_id}_{int(time.time())}.jsonl")
        return IdentityEngine(icfg, gallery=gallery, log_writer=writer)


class MemoryGallery:
    def build(self, identity_cfg):
        from poi_localization.identity.gallery.memory import InMemoryGallery
        return InMemoryGallery()


def register(reg) -> None:
    reg.register("identity_resolver", "none", NoIdentity)
    reg.register("identity_resolver", "multicue_v1", MulticueV1)
    reg.register("identity_gallery", "memory", MemoryGallery)
