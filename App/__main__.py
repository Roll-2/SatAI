"""SatAi desktop app entry point (``python -m App``).

A fixed-size (~430x680) non-resizable window that opens on the right half
of the screen, with the native macOS traffic lights. The UI (``ui/index.html``)
talks to :class:`App.bridge.AppBridge` via ``window.pywebview.api``.
"""

from __future__ import annotations

from pathlib import Path

import webview

from App.bridge import AppBridge

UI_PATH = Path(__file__).resolve().parent / "ui" / "index.html"

WINDOW_W, WINDOW_H = 430, 680
RIGHT_MARGIN = 24


def _right_half_origin() -> tuple[int, int] | None:
    """Origin (x, y) placing the window inside the right screen edge."""
    try:  # pyobjc ships with pywebview on macOS
        from AppKit import NSScreen

        frame = NSScreen.mainScreen().frame()
        x = int(frame.origin.x + frame.size.width - WINDOW_W - RIGHT_MARGIN)
        y = int(frame.origin.y + (frame.size.height - WINDOW_H) / 2)
        return max(x, 0), max(y, 0)
    except Exception:  # noqa: BLE001 — fall back to OS default position
        return None


def main() -> None:
    """Create the window and run the app loop (blocking)."""
    if not UI_PATH.exists():
        raise SystemExit(f"UI non trovata: {UI_PATH}")

    origin = _right_half_origin()
    window = webview.create_window(
        title="SatAi",
        url=str(UI_PATH),
        js_api=AppBridge(),
        width=WINDOW_W,
        height=WINDOW_H,
        x=origin[0] if origin else None,
        y=origin[1] if origin else None,
        resizable=False,          # predetermined size, per spec
        min_size=(WINDOW_W, WINDOW_H),   # no max_size in pywebview; resizable=False pins it
        background_color="#0b0f1a",
    )
    # debug=True opens Safari Web Inspector — handy during development.
    webview.start(debug=False)


if __name__ == "__main__":
    main()
