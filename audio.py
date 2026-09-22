"""Sound effects, offline and low-latency.

Uses sounddevice (CoreAudio) rather than pygame: OpenCV's macOS wheels bundle
their own libSDL2, and loading pygame's copy in the same process produces
duplicate Objective-C class registrations ("mysterious crashes") -- not a risk
worth taking on a rig that must run unattended for three hours.

Every failure path degrades to silence. The demo never stops for audio.
"""
from __future__ import annotations

import logging
import os
import threading
import wave

import numpy as np

log = logging.getLogger("invisible.audio")

SAMPLE_RATE = 44100
SOUNDS = ("pop", "detect", "fanfare", "confused")


class Audio:
    """Tiny mixer: several short sounds may overlap without cutting each other."""

    def __init__(self, sound_dir: str = "assets/sounds", enabled: bool = True,
                 volume: float = 0.8) -> None:
        self.enabled = enabled
        self.volume = float(volume)
        self._clips: dict[str, np.ndarray] = {}
        self._voices: list[list] = []          # [samples, cursor]
        self._lock = threading.Lock()
        self._stream = None

        if not enabled:
            return
        for name in SOUNDS:
            path = os.path.join(sound_dir, f"{name}.wav")
            clip = _load_wav(path)
            if clip is None:
                log.warning("sound missing: %s (continuing silently)", path)
            else:
                self._clips[name] = clip
        self._start_stream()

    def _start_stream(self) -> None:
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=512, callback=self._callback)
            self._stream.start()
        except Exception as exc:
            log.warning("audio unavailable (%s); running silent", exc)
            self._stream = None
            self.enabled = False

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        buf = np.zeros(frames, np.float32)
        with self._lock:
            keep = []
            for voice in self._voices:
                data, pos = voice
                chunk = data[pos:pos + frames]
                if chunk.size:
                    buf[:chunk.size] += chunk
                voice[1] = pos + frames
                if voice[1] < data.size:
                    keep.append(voice)
            self._voices = keep
        np.clip(buf * self.volume, -1.0, 1.0, out=buf)
        outdata[:, 0] = buf

    def play(self, name: str) -> None:
        if not self.enabled or self._stream is None:
            return
        clip = self._clips.get(name)
        if clip is None:
            return
        with self._lock:
            # Cap concurrent voices so a stuck loop can never pile up.
            if len(self._voices) > 8:
                self._voices = self._voices[-4:]
            self._voices.append([clip, 0])

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def _load_wav(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        with wave.open(path, "rb") as w:
            n, sw, ch = w.getnframes(), w.getsampwidth(), w.getnchannels()
            raw = w.readframes(n)
            rate = w.getframerate()
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
        if dtype is None:
            return None
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data /= float(np.iinfo(dtype).max if sw > 1 else 128.0)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        if rate != SAMPLE_RATE:
            idx = np.linspace(0, data.size - 1, int(data.size * SAMPLE_RATE / rate))
            data = np.interp(idx, np.arange(data.size), data).astype(np.float32)
        return np.ascontiguousarray(data, np.float32)
    except Exception as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
