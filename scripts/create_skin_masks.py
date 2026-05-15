"""
Hand skin mask annotation tool.

Controls:
    Left drag  (Draw mode)  - paint skin
    Left drag  (Erase mode) - erase
    Left drag  (Pan mode)   - pan image
    Scroll wheel            - zoom in / out (all modes)
    Tab                     - cycle mode: Draw → Erase → Pan
    H                       - initialize mask from HSV
    C                       - clear mask
    + / -                   - brush size
    S                       - save mask
    N / Space               - save (if modified) and next image
    P                       - save (if modified) and previous image
    Q / Esc                 - quit
"""

import argparse
import os
import sys
from enum import Enum, auto

import cv2
import numpy as np

TARGET_SIZE = (400, 400)
OVERLAY_ALPHA = 0.4
DEFAULT_BRUSH = 15
WINDOW = "Mask Creator"


class Mode(Enum):
    DRAW = auto()
    ERASE = auto()
    PAN = auto()

    def next(self) -> 'Mode':
        members = list(Mode)
        return members[(members.index(self) + 1) % len(members)]


_MODE_COLOR = {
    Mode.DRAW:  (180,  80,  30),
    Mode.ERASE: ( 30,  30, 180),
    Mode.PAN:   ( 30, 160,  30),
}
_MODE_LABEL = {
    Mode.DRAW:  "DRAW ",
    Mode.ERASE: "ERASE",
    Mode.PAN:   " PAN ",
}


def _hsv_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,
                       np.array([0, 20, 70], dtype=np.uint8),
                       np.array([20, 255, 255], dtype=np.uint8))


def _mask_path(image_path: str, output_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, stem + "_mask.png")


def _load_mask(image_path: str, output_dir: str) -> np.ndarray:
    path = _mask_path(image_path, output_dir)
    if os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return cv2.resize(m, TARGET_SIZE)
    return np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)


class _Viewport:
    def __init__(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def reset(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def _vis(self):
        return TARGET_SIZE[0] / self.zoom, TARGET_SIZE[1] / self.zoom

    def apply(self, img: np.ndarray) -> np.ndarray:
        vis_w, vis_h = self._vis()
        x1, y1 = int(self.pan_x), int(self.pan_y)
        x2 = min(TARGET_SIZE[0], x1 + int(vis_w) + 2)
        y2 = min(TARGET_SIZE[1], y1 + int(vis_h) + 2)
        return cv2.resize(img[y1:y2, x1:x2], TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    def to_image(self, sx: int, sy: int):
        vis_w, vis_h = self._vis()
        ix = int(self.pan_x + sx * vis_w / TARGET_SIZE[0])
        iy = int(self.pan_y + sy * vis_h / TARGET_SIZE[1])
        return (max(0, min(TARGET_SIZE[0] - 1, ix)),
                max(0, min(TARGET_SIZE[1] - 1, iy)))

    def screen_radius(self, image_radius: int) -> int:
        return max(1, int(image_radius * self.zoom))

    def zoom_at(self, sx: int, sy: int, factor: float):
        ix, iy = self.to_image(sx, sy)
        self.zoom = max(1.0, min(8.0, self.zoom * factor))
        vis_w, vis_h = self._vis()
        self.pan_x = ix - sx * vis_w / TARGET_SIZE[0]
        self.pan_y = iy - sy * vis_h / TARGET_SIZE[1]
        self._clamp()

    def pan(self, dx: int, dy: int):
        vis_w, vis_h = self._vis()
        self.pan_x -= dx * vis_w / TARGET_SIZE[0]
        self.pan_y -= dy * vis_h / TARGET_SIZE[1]
        self._clamp()

    def _clamp(self):
        vis_w, vis_h = self._vis()
        self.pan_x = max(0.0, min(TARGET_SIZE[0] - vis_w, self.pan_x))
        self.pan_y = max(0.0, min(TARGET_SIZE[1] - vis_h, self.pan_y))


class _State:
    def __init__(self):
        self.brush = DEFAULT_BRUSH
        self.mode = Mode.DRAW
        self.viewport = _Viewport()
        self.mask: np.ndarray = None
        self.dirty = False          # True once mask has been modified this image
        self.mouse_px = 0
        self.mouse_py = 0
        self._last_px = 0
        self._last_py = 0

    def cycle_mode(self):
        self.mode = self.mode.next()

    def _panel(self, x: int, y: int):
        px = x - TARGET_SIZE[0] if x >= TARGET_SIZE[0] else x
        return (max(0, min(TARGET_SIZE[0] - 1, px)),
                max(0, min(TARGET_SIZE[1] - 1, y)))

    def mouse(self, event, x, y, flags, param):
        px, py = self._panel(x, y)
        self.mouse_px = px
        self.mouse_py = py

        if self.mask is None:
            self._last_px, self._last_py = px, py
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            self.viewport.zoom_at(px, py, 1.25 if flags > 0 else 1 / 1.25)

        elif self.mode in (Mode.DRAW, Mode.ERASE):
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    ix, iy = self.viewport.to_image(px, py)
                    color = 255 if self.mode == Mode.DRAW else 0
                    cv2.circle(self.mask, (ix, iy), self.brush, color, -1)
                    self.dirty = True

        elif self.mode == Mode.PAN:
            if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                self.viewport.pan(px - self._last_px, py - self._last_py)

        self._last_px, self._last_py = px, py


def _build_display(state: _State, image: np.ndarray,
                   idx: int, total: int, image_path: str, saved: bool) -> np.ndarray:
    vp = state.viewport

    overlay = image.copy()
    overlay[state.mask > 0] = (0, 200, 0)
    blended = cv2.addWeighted(image, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
    left = vp.apply(blended)
    right = cv2.cvtColor(vp.apply(state.mask), cv2.COLOR_GRAY2BGR)

    if state.mode in (Mode.DRAW, Mode.ERASE):
        r = vp.screen_radius(state.brush)
        cx, cy = state.mouse_px, state.mouse_py
        color = (0, 120, 255) if state.mode == Mode.DRAW else (0, 0, 220)
        cv2.circle(left,  (cx, cy), r, color, 1, cv2.LINE_AA)
        cv2.circle(right, (cx, cy), r, color, 1, cv2.LINE_AA)

    display = np.hstack([left, right])

    cv2.rectangle(display, (4, 4), (70, 24), _MODE_COLOR[state.mode], -1)
    cv2.putText(display, _MODE_LABEL[state.mode], (7, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    if vp.zoom > 1.0:
        cv2.putText(display, f"{vp.zoom:.1f}x", (78, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 0), 1)

    if saved:
        cv2.putText(display, "SAVED", (display.shape[1] - 65, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

    h = display.shape[0]
    cv2.rectangle(display, (0, h - 36), (display.shape[1], h), (30, 30, 30), -1)
    info = f"[{idx+1}/{total}] {os.path.basename(image_path)}  brush={state.brush}"
    keys = "Tab=mode  H=hsv  C=clear  S=save  N/Space=next  P=prev  +/-=brush  scroll=zoom  Q=quit"
    cv2.putText(display, info, (6, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(display, keys, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return display


def _collect_images(path: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if os.path.isfile(path):
        return [path]
    images = []
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                images.append(os.path.join(root, f))
    images.sort()
    return images


def _save(state: _State, image_path: str, output_dir: str):
    cv2.imwrite(_mask_path(image_path, output_dir), state.mask)
    print(f"Saved: {_mask_path(image_path, output_dir)}")
    state.dirty = False


def run(images_dir: str, output_dir: str):
    images = _collect_images(images_dir)
    if not images:
        print(f"No images found in: {images_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(images)} images. Masks saved to: {output_dir}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, TARGET_SIZE[0] * 2, TARGET_SIZE[1])
    state = _State()
    cv2.setMouseCallback(WINDOW, state.mouse)

    idx = 0
    while 0 <= idx < len(images):
        image_path = images[idx]
        raw = cv2.imread(image_path)
        if raw is None:
            print(f"Cannot read: {image_path}, skipping")
            idx += 1
            continue

        image = cv2.resize(raw, TARGET_SIZE)
        state.mask = _load_mask(image_path, output_dir)
        state.dirty = False
        state.viewport.reset()
        saved = os.path.exists(_mask_path(image_path, output_dir))

        while True:
            display = _build_display(state, image, idx, len(images), image_path, saved)
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                return
            elif key == 9:                              # Tab — cycle mode
                state.cycle_mode()
            elif key == ord('h'):
                state.mask = _hsv_mask(image)
                state.dirty = True
                saved = False
            elif key == ord('c'):
                state.mask = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
                state.dirty = True
                saved = False
            elif key == ord('s'):
                _save(state, image_path, output_dir)
                saved = True
            elif key in (ord('n'), ord(' ')):
                if state.dirty:
                    _save(state, image_path, output_dir)
                    saved = True
                idx += 1
                break
            elif key == ord('p'):
                if state.dirty:
                    _save(state, image_path, output_dir)
                    saved = True
                idx -= 1
                break
            elif key in (ord('+'), ord('=')):
                state.brush = min(60, state.brush + 2)
            elif key == ord('-'):
                state.brush = max(2, state.brush - 2)

    cv2.destroyAllWindows()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Hand skin mask annotation tool")
    parser.add_argument("images_dir", nargs="?", default=".",
                        help="Directory with hand images")
    parser.add_argument("--output", default="skin_masks",
                        help="Output directory for masks (default: skin_masks/)")
    args = parser.parse_args()
    run(args.images_dir, args.output)


if __name__ == "__main__":
    main()
