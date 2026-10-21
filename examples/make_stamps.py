#!/usr/bin/env python3
"""
Генератор демо-штампов для примеров Splotch.

Каждый штамп - карта высот вида на рабочую грань:
  чёрный (0)  - материала нет (задаёт форму штампа);
  серый       - основание рабочей грани;
  светлее     - выступы (на сургуче станут углублениями);
  темнее      - углубления (на сургуче станут выпуклыми).
Текст рисуется зеркально, как на настоящем штампе.

  python3 examples/make_stamps.py examples/stamps
"""

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

PX = 1000
C = PX / 2
BASE = 150      # основание грани
HIGH = 255      # выступ
LOW = 70        # углубление


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def poly(n, r, rot=0.0, cx=C, cy=C):
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n), cy + r * math.sin(rot + 2 * math.pi * k / n))
            for k in range(n)]


def star(n, r_out, r_in, rot=-math.pi / 2, cx=C, cy=C):
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = rot + math.pi * k / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def heart(scale, cx=C, cy=C, steps=400):
    pts = []
    for k in range(steps):
        t = 2 * math.pi * k / steps
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * scale, cy - (y + 2.5) * scale))
    return pts


def inset(points, k, cx=C, cy=C):
    return [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in points]


def mirrored_text(img, text, xy, size, fill):
    layer = Image.new("L", img.size, 0)
    ImageDraw.Draw(layer).text(xy, text, font=font(size), fill=255, anchor="mm")
    layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img).copy()
    arr[np.asarray(layer) > 128] = fill
    return Image.fromarray(arr)


def finish(img, outline, path, bevel=10, blur=2.0):
    """Сгладить рельеф и сделать фаску по контуру штампа."""
    arr = ndimage.gaussian_filter(np.asarray(img).astype(float), blur)
    mask = np.asarray(outline) > 0
    ramp = np.clip(ndimage.distance_transform_edt(mask) / bevel, 0, 1)
    arr = arr * (0.35 + 0.65 * ramp)
    arr[~mask] = 0
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def new(shape_pts=None, ellipse=None):
    img = Image.new("L", (PX, PX), 0)
    outline = Image.new("L", (PX, PX), 0)
    for target, fill in ((img, BASE), (outline, 255)):
        d = ImageDraw.Draw(target)
        if ellipse:
            d.ellipse(ellipse, fill=fill)
        else:
            d.polygon(shape_pts, fill=fill)
    return img, outline, ImageDraw.Draw(img)


def stamp_circle(path):
    R = 480
    img, outline, d = new(ellipse=[C - R, C - R, C + R, C + R])
    d.ellipse([C - R + 10, C - R + 10, C + R - 10, C + R - 10], outline=HIGH, width=28)
    for k in range(48):
        a = 2 * math.pi * k / 48
        x, y = C + 405 * math.cos(a), C + 405 * math.sin(a)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=HIGH)
    d.ellipse([C - 360, C - 360, C + 360, C + 360], outline=HIGH, width=10)
    # лавровый венок из листочков
    for side in (-1, 1):
        for k in range(9):
            a = math.radians(115 + k * 13) if side < 0 else math.radians(65 - k * 13)
            x, y = C + 300 * math.cos(a), C + 300 * math.sin(a)
            leaf = poly(24, 1)
            ang = a + side * math.pi / 2.6
            leaf = [(x + 34 * math.cos(ang) * px - 13 * math.sin(ang) * py,
                     y + 34 * math.sin(ang) * px + 13 * math.cos(ang) * py)
                    for px, py in [((p[0] - C), (p[1] - C)) for p in leaf]]
            d.polygon(leaf, fill=HIGH)
    img = mirrored_text(img, "S", (C, C - 15), 420, LOW)
    finish(img, outline, path)


def stamp_star(path):
    pts = star(5, 490, 215, cx=C, cy=C + 30)
    img, outline, d = new(pts)
    d.line(inset(pts, 0.86, C, C + 30) + [inset(pts, 0.86, C, C + 30)[0]], fill=HIGH, width=18, joint="curve")
    d.polygon(star(5, 150, 66, cx=C, cy=C + 30), fill=LOW)
    for k in range(5):
        a = -math.pi / 2 + 2 * math.pi * k / 5
        for j, r in enumerate((230, 300, 360)):
            x, y = C + r * math.cos(a), C + 30 + r * math.sin(a)
            s = 22 - 5 * j
            d.ellipse([x - s, y - s, x + s, y + s], fill=HIGH)
    finish(img, outline, path)


def stamp_diamond(path):
    pts = [(C, 20), (980, C), (C, 980), (20, C)]
    img, outline, d = new(pts)
    for k, w in ((0.88, 18), (0.62, 10)):
        p = inset(pts, k)
        d.line(p + [p[0]], fill=HIGH, width=w, joint="curve")
    # решётка-ромбики внутри
    for k in range(-3, 4):
        for j in range(-3, 4):
            if abs(k) + abs(j) > 3:
                continue
            x, y = C + k * 75, C + j * 75
            q = [(x, y - 26), (x + 26, y), (x, y + 26), (x - 26, y)]
            d.polygon(q, fill=LOW if (k + j) % 2 else HIGH)
    for p in inset(pts, 0.75):
        d.ellipse([p[0] - 20, p[1] - 20, p[0] + 20, p[1] + 20], fill=HIGH)
    finish(img, outline, path)


def stamp_hexagon(path):
    pts = poly(6, 490, rot=0)
    img, outline, d = new(pts)
    p = inset(pts, 0.9)
    d.line(p + [p[0]], fill=HIGH, width=22, joint="curve")
    # соты
    r = 62
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = C + i * r * 1.5
            y = C + (j + 0.5 * (i % 2)) * r * math.sqrt(3)
            if math.hypot(x - C, y - C) > 330:
                continue
            d.polygon(poly(6, r - 9, cx=x, cy=y), fill=LOW)
    # пчела в центре
    d.polygon(poly(6, r * 1.6, cx=C, cy=C), fill=BASE)
    d.ellipse([C - 46, C - 70, C + 46, C + 75], fill=HIGH)
    for yy in (-25, 15):
        d.rectangle([C - 46, C + yy, C + 46, C + yy + 16], fill=LOW)
    for side in (-1, 1):
        d.ellipse([C + side * 30 - 38, C - 95, C + side * 30 + 38, C - 30], outline=HIGH, width=9)
    finish(img, outline, path)


def stamp_heart(path):
    pts = heart(28)
    img, outline, d = new(pts)
    d.line(inset(pts, 0.88) + [inset(pts, 0.88)[0]], fill=HIGH, width=16, joint="curve")
    d.polygon(heart(9, cy=C - 20), fill=LOW)
    for k in range(-3, 4):
        x = C + k * 70
        y = C + 120 - abs(k) * 22
        d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=HIGH)
    finish(img, outline, path)


def stamp_shield(path):
    pts = [(120, 60), (880, 60), (880, 480)]
    for k in range(1, 40):
        t = k / 40
        pts.append((880 - 380 * t, 480 + 470 * math.sin(t * math.pi / 2)))
    for k in range(1, 40):
        t = k / 40
        pts.append((C - 380 * t, 950 - 470 * (1 - math.cos(t * math.pi / 2))))
    pts.append((120, 480))
    img, outline, d = new(pts)
    p = inset(pts, 0.88, C, 470)
    d.line(p + [p[0]], fill=HIGH, width=16, joint="curve")
    d.rectangle([C - 18, 140, C + 18, 820], fill=HIGH)
    d.rectangle([180, 380, 820, 416], fill=HIGH)
    d.polygon(star(4, 70, 25, rot=0, cx=330, cy=255), fill=LOW)
    d.polygon(star(4, 70, 25, rot=0, cx=670, cy=255), fill=LOW)
    d.polygon(poly(3, 70, rot=-math.pi / 2, cx=330, cy=590), fill=LOW)
    d.polygon(poly(3, 70, rot=-math.pi / 2, cx=670, cy=590), fill=LOW)
    finish(img, outline, path)


STAMPS = {
    "circle": stamp_circle,
    "star": stamp_star,
    "diamond": stamp_diamond,
    "hexagon": stamp_hexagon,
    "heart": stamp_heart,
    "shield": stamp_shield,
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "stamps"
    os.makedirs(out, exist_ok=True)
    for name, fn in STAMPS.items():
        fn(os.path.join(out, name + ".png"))
        print(os.path.join(out, name + ".png"))
