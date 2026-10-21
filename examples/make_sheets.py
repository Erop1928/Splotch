#!/usr/bin/env python3
"""
Собирает картинки для README из результатов в examples/*/.

  python3 examples/make_sheets.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from splotch import save_preview  # noqa: E402

FONT = ImageFont.load_default(size=22)
INK = (90, 60, 50)


def ex(*p):
    return os.path.join(HERE, *p)


def height(prefix):
    meta = json.load(open(prefix + ".json"))
    raw = np.asarray(Image.open(prefix + "_height.png")).astype(float) / 65535
    return raw * meta["heightmap"]["z_mm_at_white"], meta["resolution_mm"]


def hires(prefix, zoom=2.5):
    """Превью с повышенным разрешением (детали пузырьков видны лучше)."""
    h, dx = height(prefix)
    out = prefix + "_hires.png"
    save_preview(out, ndimage.zoom(h, zoom, order=3), dx / zoom)
    img = Image.open(out).convert("RGB")
    os.remove(out)
    return img


def paper():
    return Image.open(ex("gallery", "circle_preview.png")).convert("RGB").getpixel((2, 2))


def tile(img, label, size, stamp=None):
    t = Image.new("RGB", (size, size + 36), paper())
    s = min(1, (size - 16) / max(img.size))
    img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    t.paste(img, ((size - img.width) // 2, (size - img.height) // 2 + 4))
    if stamp:
        st = Image.open(stamp).convert("L").resize((72, 72), Image.LANCZOS)
        t.paste(Image.merge("RGB", [st] * 3), (size - 80, 8))
    ImageDraw.Draw(t).text((size // 2, size + 12), label, font=FONT, fill=INK, anchor="mm")
    return t


def sheet(tiles, cols, out):
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    o = Image.new("RGB", (cols * w, rows * h), paper())
    for i, t in enumerate(tiles):
        o.paste(t, ((i % cols) * w, (i // cols) * h))
    o.save(out, optimize=True)
    print(out)


def underfill_map(with_air, without_air):
    """Где запертый воздух не пустил сургуч: красным поверх серого рельефа."""
    a, dx = height(with_air)
    b, _ = height(without_air)
    hh, ww = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:hh, :ww], b[:hh, :ww]
    d = np.clip((b - a) / 0.15, 0, 1)
    gy, gx = np.gradient(a, dx)
    shade = np.clip((-gx * 0.5 + gy * 0.5 + 0.7) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    base = 0.35 + 0.5 * shade
    rgb = np.stack([base + (1 - base) * d, base * (1 - 0.8 * d), base * (1 - 0.8 * d)], -1)
    rgb[a <= 0.02] = np.array(paper()) / 255.0
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return img.resize((img.width * 2, img.height * 2), Image.LANCZOS)


if __name__ == "__main__":
    names = ["circle", "star", "diamond", "hexagon", "heart", "shield"]
    sheet([tile(Image.open(ex("gallery", n + "_preview.png")), n, 360, ex("stamps", n + ".png"))
           for n in names], 3, ex("gallery.png"))

    cond = [("1_classic", "classic"), ("2_high_pour", "high pour, offset, tilt"),
            ("3_late_press", "pressed too late"), ("4_little_wax", "little wax, weak press")]
    sheet([tile(Image.open(ex("conditions", n + "_preview.png")), label, 360) for n, label in cond],
          4, ex("conditions.png"))

    sheet([tile(underfill_map(ex("conditions", "1_classic"), ex("bubbles", "1_no_air")),
                "trapped air (red: unfilled)", 440),
           tile(hires(ex("bubbles", "3_pinholes")), "pinholes", 440),
           tile(hires(ex("bubbles", "4_big_bubbles")), "big bubbles, hot press", 440)],
          3, ex("air_and_bubbles.png"))
