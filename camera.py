"""Threaded capture that always keeps the latest frame (low-lag RTSP/webcam)."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class LatestFrameStream:
    """
    Background reader that overwrites the newest frame instead of queueing.

    Prevents OpenCV buffer backlog when inference is slower than the camera FPS.
    """

    def __init__(self, source: Union[str, int], reconnect_sec: float = 3.0) -> None:
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.reconnect_sec = reconnect_sec
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ok = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "LatestFrameStream":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="LatestFrameStream", daemon=True)
        self._thread.start()
        return self

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        # Shrink internal buffer when backend supports it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        return cap

    def _loop(self) -> None:
        while self._running:
            try:
                cap = self._open()
            except RuntimeError:
                time.sleep(self.reconnect_sec)
                continue

            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    time.sleep(self.reconnect_sec)
                    break
                with self._lock:
                    self._frame = frame
                    self._ok = True
            cap.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
