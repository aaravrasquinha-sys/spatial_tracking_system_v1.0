"""Adapters: today's implementations behind each slot (sts/slots.py).

Rule 2 of docs/ARCHITECTURE.md: existing code is WRAPPED, not refactored. An adapter
may build a command line, read a config, or construct an object from a legacy module;
it must not contain perception/SLAM/localization maths.
"""
from __future__ import annotations


def register_all(reg) -> None:
    from sts.adapters import offline, runtime
    offline.register(reg)
    runtime.register(reg)
