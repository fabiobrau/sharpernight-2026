"""Session-only top-5 invisibility times.

Deliberately in memory only. Nothing about a child who walked past a camera
should outlive the afternoon, so there is no load() and no save().
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Entry:
    seconds: float
    at: float = field(default_factory=time.time)


class Scoreboard:
    def __init__(self, size: int = 5, min_seconds: float = 1.0) -> None:
        self.size = size
        self.min_seconds = min_seconds
        self.entries: list[Entry] = []
        self.attempts = 0

    def submit(self, seconds: float) -> bool:
        """Record a run. Returns True if it landed on the board."""
        if seconds < self.min_seconds:
            return False
        self.attempts += 1
        self.entries.append(Entry(seconds))
        self.entries.sort(key=lambda e: e.seconds, reverse=True)
        del self.entries[self.size:]
        return any(abs(e.seconds - seconds) < 1e-9 for e in self.entries)

    def is_new_best(self, seconds: float) -> bool:
        return bool(self.entries) and abs(self.entries[0].seconds - seconds) < 1e-9

    @property
    def best(self) -> float:
        return self.entries[0].seconds if self.entries else 0.0

    def top(self) -> list[Entry]:
        return list(self.entries)

    def reset(self) -> None:
        self.entries.clear()
        self.attempts = 0
