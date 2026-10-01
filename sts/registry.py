"""Slot registry: name -> implementation factory. No global magic: adapters register
themselves explicitly in sts.adapters.register_all(), and tests can build a fresh
registry with a fake implementation swapped in."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from sts.slots import SLOTS


class SlotError(KeyError):
    pass


class SlotRegistry:
    def __init__(self):
        self._impls: Dict[str, Dict[str, Callable[..., Any]]] = {s: {} for s in SLOTS}

    def register(self, slot: str, name: str, factory: Callable[..., Any]) -> None:
        if slot not in SLOTS:
            raise SlotError(f"unknown slot {slot!r}; known: {sorted(SLOTS)}")
        if name in SLOTS[slot].reserved:
            raise SlotError(f"{slot}:{name} is a RESERVED name (planned, not built); register it under "
                            f"that name only when it is real and remove it from sts/slots.py `reserved`")
        self._impls[slot][name] = factory

    def names(self, slot: str) -> List[str]:
        return sorted(self._impls[slot])

    def resolve(self, slot: str, chosen: Optional[str] = None) -> Callable[..., Any]:
        if slot not in SLOTS:
            raise SlotError(f"unknown slot {slot!r}")
        name = chosen or SLOTS[slot].default
        if name in SLOTS[slot].reserved:
            raise SlotError(f"slot {slot!r}: implementation {name!r} is reserved (planned, not built yet). "
                            f"Built: {self.names(slot)}")
        try:
            return self._impls[slot][name]
        except KeyError:
            raise SlotError(f"slot {slot!r}: no implementation {name!r}. Built: {self.names(slot)}") from None

    def create(self, slot: str, chosen: Optional[str] = None, *a, **kw):
        return self.resolve(slot, chosen)(*a, **kw)

    def default_complete(self) -> List[str]:
        """Slots whose DEFAULT implementation is not registered (a packaging bug)."""
        return [s for s, spec in SLOTS.items() if spec.default not in self._impls[s]]


_DEFAULT: Optional[SlotRegistry] = None


def default_registry() -> SlotRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        from sts.adapters import register_all
        reg = SlotRegistry()
        register_all(reg)
        _DEFAULT = reg
    return _DEFAULT
