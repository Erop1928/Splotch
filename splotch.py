#!/usr/bin/env python3
"""
Splotch - физическая (2.5D) симуляция сургучной печати.
Капнул, прижал - получил STL.

Этапы:
  1. Налив: струя сургуча падает с заданной высоты, укладывается витками,
     растекается под действием силы тяжести (тонкоплёночное приближение,
     вязкость зависит от температуры), остывает. При большой высоте - брызги.
  2. Пауза перед прижимом (сургуч остывает и густеет).
  3. Прижим: штамп (карта высот) опускается со смещением/поворотом/наклоном.
     Вытесняемый сургуч перетекает под давлением (уравнение Пуассона для
     давления в зазоре, проводимость ~ толщина^3) в полости штампа и наружу,
     образуя валик. Глубина определяется балансом силы прижима и
     сопротивления пластичного сургуча (формула Прандтля для сжатия слоя).
  4. Выдержка под штампом (сургуч остывает быстрее от холодного штампа),
     подъём штампа.
  5. Экспорт: STL и/или 16-битная карта высот PNG (+ JSON с масштабом, превью).

Формат карты высот штампа:
  яркость = высота материала штампа (выступ к сургучу);
  цвет "пусто" (по умолчанию чёрный, --empty white для белого) или прозрачность
  = отсутствие материала. Контур штампа = область, где материал есть
  (замкнутые "пустые" области внутри контура считаются самым глубоким
  углублением рельефа, если не указан --holes-through).
  Карта считается видом НА РАБОЧУЮ ГРАНЬ штампа, поэтому оттиск зеркалится
  (отключается --no-mirror).

Параметры с рандомизацией (любой числовой параметр симуляции):
  "40"      - фиксированное значение
  "30:50"   - равномерно случайное в диапазоне
  "40~5"    - нормальное распределение (среднее~сигма)

Пример:
  python3 splotch.py --stamp stamp.png --stamp-size 25 --relief-depth 1.2 \
      --volume 1.5 --drop-height 30:80 --force 40~10 --offset-x -1:1 \
      --offset-y -1:1 --rotation 0~3 --tilt 0:2 --count 5 --out seals/seal
  python3 splotch.py --make-demo-stamp demo_stamp.png
"""

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

G_MM = 9810.0           # мм/с^2
RHO = 1.1e-6            # кг/мм^3 (плотность сургуча ~1.1 г/см^3)
T_REF = 120.0           # °C, температура, к которой привязаны вязкость и предел текучести


# ----------------------------------------------------------------------------
# Рандомизируемые параметры
# ----------------------------------------------------------------------------

class Spec:
    """Значение параметра: число, диапазон a:b или нормальное m~s."""

    def __init__(self, text):
        self.text = str(text).strip()
        t = self.text
        try:
            if "~" in t:
                m, s = t.split("~")
                self.kind, self.a, self.b = "normal", float(m), float(s)
            elif ":" in t:
                a, b = t.split(":")
                self.kind, self.a, self.b = "uniform", float(a), float(b)
            else:
                self.kind, self.a, self.b = "fixed", float(t), 0.0
        except ValueError:
            raise argparse.ArgumentTypeError(f"не понимаю значение '{t}' (число, a:b или m~s)")

    def sample(self, rng):
        if self.kind == "normal":
            return float(rng.normal(self.a, self.b))
        if self.kind == "uniform":
            return float(rng.uniform(self.a, self.b))
        return self.a

    def __repr__(self):
        return self.text


# Параметры симуляции: имя -> (значение по умолчанию, мин, макс, описание)
SIM_PARAMS = {
    "volume":        ("1.5", 0.05, 50, "объём сургуча, см^3 (мл)"),
    "drop_height":   ("40", 0, 2000, "высота налива (падения струи), мм"),
    "pour_rate":     ("400", 10, 1e5, "скорость налива, мм^3/с"),
    "pour_wobble":   ("0.8", 0, 20, "дрожание руки при наливе, мм"),
    "stream_radius": ("1.5", 0.2, 10, "радиус струи, мм"),
    "temp":          ("115", 60, 250, "температура сургуча при наливе, °C"),
    "ambient":       ("22", -20, 60, "температура бумаги/воздуха, °C"),
    "delay":         ("4", 0, 600, "пауза между концом налива и прижимом, с"),
    "force":         ("40", 0.1, 5000, "сила прижима, Н"),
    "offset_x":      ("0", -100, 100, "смещение штампа от центра лужицы по X, мм"),
    "offset_y":      ("0", -100, 100, "смещение штампа от центра лужицы по Y, мм"),
    "rotation":      ("0", -360, 360, "поворот штампа, градусы"),
    "tilt":          ("0", 0, 30, "наклон штампа, градусы"),
    "tilt_dir":      ("0:360", -360, 720, "направление наклона, градусы"),
    "press_speed":   ("4", 0.05, 100, "скорость опускания штампа, мм/с"),
    "dwell":         ("5", 0, 600, "выдержка под штампом, с"),
    "viscosity":     ("15", 0.1, 1e5, "вязкость при 120°C, Па*с"),
    "visc_slope":    ("0.07", 0, 1, "рост ln(вязкости) на 1°C остывания"),
    "yield_stress":  ("250", 1, 1e7, "предел текучести при 120°C, Па"),
    "yield_slope":   ("0.09", 0, 1, "рост ln(предела текучести) на 1°C остывания"),
    "cool_time":     ("5", 0.1, 1000, "время остывания слоя 1 мм на бумаге, с"),
    "stamp_chill":   ("3", 1, 50, "во сколько раз быстрее остывает сургуч под штампом"),
}


# ----------------------------------------------------------------------------
# Штамп
# ----------------------------------------------------------------------------

def load_stamp(path, empty="black", threshold=0.02, holes_through=False, blur_px=0.0):
    im = Image.open(path)
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        alpha = np.asarray(rgba)[..., 3].astype(np.float64) / 255.0
        val = np.asarray(rgba.convert("L")).astype(np.float64) / 255.0
    elif im.mode.startswith("I"):
        arr = np.asarray(im).astype(np.float64)
        val = arr / (65535.0 if arr.max() > 255 else 255.0)
    elif im.mode == "F":
        arr = np.asarray(im).astype(np.float64)
        val = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-12)
    else:
        val = np.asarray(im.convert("L")).astype(np.float64) / 255.0
    if empty == "white":
        val = 1.0 - val
    material = val > threshold
    if alpha is not None:
        material &= alpha > 0.5
        val = np.where(alpha > 0.5, val, 0.0)
    outline = material if holes_through else ndimage.binary_fill_holes(material)
    height = np.where(outline, val, 0.0)
    if blur_px > 0:
        height = ndimage.gaussian_filter(height, blur_px)
    if not outline.any():
        raise SystemExit("В карте штампа нет материала - проверьте --empty / --empty-threshold")
    return height, outline


def stamp_ceiling(X, Y, stamp, size_mm, ox, oy, rot_deg, tilt_deg, tilt_dir_deg,
                  mirror, depth, z, cache=None):
    """Высота нижней поверхности штампа над бумагой. inf - штампа нет."""
    if cache is None or "rel" not in cache:
        height, outline = stamp
        hgt, wid = height.shape
        ppm = wid / size_mm
        dx = X - ox
        dy = Y - oy
        r = math.radians(rot_deg)
        u = math.cos(r) * dx + math.sin(r) * dy
        v = -math.sin(r) * dx + math.cos(r) * dy
        if mirror:
            u = -u
        col = u * ppm + (wid - 1) / 2.0
        row = -v * ppm + (hgt - 1) / 2.0
        s = ndimage.map_coordinates(height, [row, col], order=1, cval=0.0)
        o = ndimage.map_coordinates(outline.astype(np.float64), [row, col], order=1, cval=0.0) > 0.5
        td = math.radians(tilt_dir_deg)
        tilt = math.tan(math.radians(tilt_deg)) * (dx * math.cos(td) + dy * math.sin(td))
        rel = (1.0 - np.clip(s, 0, 1)) * depth + tilt
        rel = np.where(o, rel, np.inf)
        if cache is not None:
            cache["rel"] = rel
    else:
        rel = cache["rel"]
    return rel + z


def make_demo_stamp(path, px=1000):
    """Круглый штамп: основание 50% серого, бортик и рисунок - выступы (белый)."""
    im = Image.new("L", (px, px), 0)
    d = ImageDraw.Draw(im)
    c = px / 2
    R = px * 0.49
    d.ellipse([c - R, c - R, c + R, c + R], fill=128)                 # основание
    w = px * 0.035
    d.ellipse([c - R, c - R, c + R, c + R], outline=255, width=int(w))  # бортик
    r2 = R * 0.80
    d.ellipse([c - r2, c - r2, c + r2, c + r2], outline=230, width=int(px * 0.012))
    # звезда
    pts = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        rr = R * (0.42 if k % 2 == 0 else 0.17)
        pts.append((c + rr * math.cos(a), c - 0.12 * px - rr * math.sin(a)))
    d.polygon(pts, fill=255)
    # текст (на рабочей грани штампа текст зеркальный)
    try:
        font = ImageFont.load_default(size=int(px * 0.16))
    except TypeError:
        font = ImageFont.load_default()
    txt = Image.new("L", (px, px), 0)
    td = ImageDraw.Draw(txt)
    td.text((c, c + 0.25 * px), "WAX", font=font, fill=255, anchor="mm")
    txt = txt.transpose(Image.FLIP_LEFT_RIGHT)
    im = Image.fromarray(np.maximum(np.asarray(im), np.asarray(txt)))
    # мелкие точки по кругу
    d = ImageDraw.Draw(im)
    for k in range(36):
        a = 2 * math.pi * k / 36
        rr = R * 0.88
        x, y = c + rr * math.cos(a), c + rr * math.sin(a)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=210)
    im = im.filter(ImageFilter.GaussianBlur(1.5))
    # контур восстанавливаем чётким (фон строго 0)
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse([c - R, c - R, c + R, c + R], fill=255)
    out = np.where(np.asarray(mask) > 0, np.maximum(np.asarray(im), 20), 0).astype(np.uint8)
    Image.fromarray(out).save(path)
    print(f"Демо-штамп сохранён: {path} (рекомендуемый --stamp-size 25)")


# ----------------------------------------------------------------------------
# Физика сургуча
# ----------------------------------------------------------------------------

class Wax:
    def __init__(self, n, dx, P, rng):
        self.n, self.dx, self.P = n, dx, P
        self.h = np.zeros((n, n))                # толщина, мм
        self.T = np.full((n, n), P["ambient"])   # температура, °C
        self.contact = np.zeros((n, n), bool)    # касание штампа (быстрое остывание)
        # неоднородность растекания (шероховатость бумаги, неравномерность нагрева)
        noise = ndimage.gaussian_filter(rng.normal(size=(n, n)), 1.2 / dx)
        noise /= max(noise.std(), 1e-9)
        self.noise = np.exp(0.35 * np.clip(noise, -1.5, 1.5))
        self.h_pin = 0.3   # мм: минимальная толщина края, ниже которой плёнка не течёт

    def viscosity(self, T):
        P = self.P
        return np.minimum(P["viscosity"] * np.exp(P["visc_slope"] * (T_REF - T)), 1e9)

    def yield_stress(self, T):
        P = self.P
        return P["yield_stress"] * math.exp(P["yield_slope"] * (T_REF - T))

    def window(self, margin=14):
        rows = np.flatnonzero(self.h.any(axis=1))
        cols = np.flatnonzero(self.h.any(axis=0))
        if len(rows) == 0:
            return slice(0, 0), slice(0, 0)
        n = self.n
        return (slice(max(rows[0] - margin, 1), min(rows[-1] + margin + 1, n - 1)),
                slice(max(cols[0] - margin, 1), min(cols[-1] + margin + 1, n - 1)))

    def cool(self, sl, dt):
        h, T = self.h[sl], self.T[sl]
        tau = self.P["cool_time"] * np.maximum(h, 0.15)
        tau = np.where(self.contact[sl], tau / self.P["stamp_chill"], tau)
        Ta = self.P["ambient"]
        T[...] = Ta + (T - Ta) * np.exp(-dt / tau)

    def flow(self, duration, blocked=None, max_sub=10):
        """Гравитационное растекание: h_t = div( rho g h^3 / (3 mu) grad h )."""
        dx = self.dx
        t = 0.0
        while t < duration - 1e-12:
            sl = self.window()
            if sl[0].stop <= sl[0].start:
                return
            t_batch = 0.0
            for _ in range(max_sub):
                if t >= duration - 1e-12:
                    break
                h, T = self.h[sl], self.T[sl]
                D = (RHO * G_MM / 3.0) * h ** 3 / (self.viscosity(T) * 1e-3) * self.noise[sl]
                D[h < self.h_pin] = 0.0
                if blocked is not None:
                    D[blocked[sl]] = 0.0
                Dmax = D.max()
                if Dmax < 1e-9:
                    t_batch += duration - t
                    t = duration
                    break
                dt = min(0.24 * dx * dx / Dmax, duration - t)
                b = blocked[sl] if blocked is not None else None
                dh = np.zeros_like(h)
                dE = np.zeros_like(h)
                for axis in (0, 1):
                    a = [slice(None), slice(None)]
                    lo, hi = list(a), list(a)
                    lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                    lo, hi = tuple(lo), tuple(hi)
                    dH = h[hi] - h[lo]
                    src_lo = dH < 0
                    Df = np.where(src_lo, D[lo], D[hi])
                    Tf = np.where(src_lo, T[lo], T[hi])
                    q = -Df * dH * (dt / (dx * dx))     # объём (в мм толщины) из lo в hi
                    if b is not None:
                        q[b[lo] | b[hi]] = 0.0
                    dh[lo] -= q
                    dh[hi] += q
                    dE[lo] -= q * Tf
                    dE[hi] += q * Tf
                E = h * T + dE
                h += dh
                np.maximum(h, 0.0, out=h)
                T[...] = np.where(h > 1e-9, E / np.maximum(h, 1e-12), T)
                t += dt
                t_batch += dt
            if t_batch > 0:
                self.cool(sl, t_batch)

    def deposit(self, x, y, sigma, vol, temp, X, Y):
        """Добавить каплю объёмом vol (мм^3) гауссовой формы."""
        dx = self.dx
        n = self.n
        cj = int(round(x / dx + (n - 1) / 2))
        ci = int(round((n - 1) / 2 - y / dx))
        r = int(math.ceil(3.5 * sigma / dx)) + 1
        i0, i1 = max(ci - r, 0), min(ci + r + 1, n)
        j0, j1 = max(cj - r, 0), min(cj + r + 1, n)
        if i0 >= i1 or j0 >= j1:
            return
        g = np.exp(-((X[i0:i1, j0:j1] - x) ** 2 + (Y[i0:i1, j0:j1] - y) ** 2) / (2 * sigma ** 2))
        s = g.sum()
        if s <= 0:
            return
        add = g * (vol / (s * dx * dx))
        h = self.h[i0:i1, j0:j1]
        T = self.T[i0:i1, j0:j1]
        hn = h + add
        T[...] = np.where(hn > 1e-12, (h * T + add * temp) / np.maximum(hn, 1e-12), T)
        h[...] = hn

    # ------------------------------------------------------------------
    def redistribute(self, c, max_outer=60):
        """Вытеснить сургуч, оказавшийся выше поверхности штампа.

        Давление в зоне контакта: div(K grad p) = -источник, p=0 там, где есть
        свободное место (полости штампа, край штампа). K ~ толщина слоя^3.
        """
        h, T = self.h, self.T
        n = self.n
        stamp = np.isfinite(c)
        border = np.zeros((n, n), bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        moved = 0.0
        for _ in range(max_outer):
            ex = np.where(stamp, np.maximum(h - c, 0.0), 0.0)
            tot = ex.sum()
            if tot < 1e-7:
                break
            moved += tot
            h -= ex
            interior = stamp & (h >= c - 1e-6) & ~border
            # источник в клетке на границе сетки - просто оставляем (крайне редко)
            ii = np.flatnonzero(interior.ravel())
            m = len(ii)
            idx = -np.ones(n * n, np.int64)
            idx[ii] = np.arange(m)
            hf = h.ravel()
            Tf = T.ravel()
            exf = ex.ravel()[ii]
            Tsrc = Tf[ii]
            diag = np.zeros(m)
            rows, cols, vals = [], [], []
            bnb, bK, bsrc = [], [], []
            for off in (1, -1, n, -n):
                nb = ii + off
                K = (0.5 * (hf[ii] + hf[nb])) ** 3 + 1e-6
                diag += K
                inn = idx[nb] >= 0
                rows.append(np.arange(m)[inn])
                cols.append(idx[nb][inn])
                vals.append(-K[inn])
                out = ~inn
                bnb.append(nb[out])
                bK.append(K[out])
                bsrc.append(np.arange(m)[out])
            rows = np.concatenate(rows + [np.arange(m)])
            cols = np.concatenate(cols + [np.arange(m)])
            vals = np.concatenate(vals + [diag])
            A = csr_matrix((vals, (rows, cols)), shape=(m, m)).tocsc()
            p = spsolve(A, exf)
            bnb = np.concatenate(bnb)
            bK = np.concatenate(bK)
            bsrc = np.concatenate(bsrc)
            dv = bK * p[bsrc]
            dv = np.maximum(dv, 0.0)
            # нормировка (сохранение объёма с учётом численной погрешности)
            sdv = dv.sum()
            if sdv > 0:
                dv *= exf.sum() / sdv
            add = np.bincount(bnb, weights=dv, minlength=n * n)
            addE = np.bincount(bnb, weights=dv * Tsrc[bsrc], minlength=n * n)
            hn = hf + add
            Tf[...] = np.where(add > 0, (hf * Tf + addE) / np.maximum(hn, 1e-12), Tf)
            hf[...] = hn
        return moved


# ----------------------------------------------------------------------------
# Симуляция
# ----------------------------------------------------------------------------

def simulate(P, stamp, args, rng, log):
    t0 = time.time()
    dx = args.resolution
    V = P["volume"] * 1000.0  # мм^3
    r_puddle = math.sqrt(V / (math.pi * 2.0))
    reach = max(args.stamp_size * 0.75 + math.hypot(P["offset_x"], P["offset_y"]), r_puddle * 1.5)
    canvas = args.canvas or (2 * reach + 12 + 0.02 * P["drop_height"])
    n = int(math.ceil(canvas / dx)) | 1
    coords = (np.arange(n) - (n - 1) / 2) * dx
    X, Y = np.meshgrid(coords, -coords)
    # налив и растекание считаются на более грубой сетке (там нет мелких деталей)
    f = max(args.pour_coarsen, 1)
    nc = int(math.ceil(n / f)) | 1
    coords_c = (np.arange(nc) - (nc - 1) / 2) * dx * f
    Xc, Yc = np.meshgrid(coords_c, -coords_c)
    wax = Wax(nc, dx * f, P, rng)
    log(f"  сетка {n}x{n}, шаг {dx} мм (налив: шаг {dx * f:.3g} мм), поле {n * dx:.1f} мм")

    # --- 1. налив ---
    H = P["drop_height"]
    v_imp = math.sqrt(2 * G_MM * H)                            # мм/с
    T_arr = P["temp"] - 0.03 * H                               # остывание струи в полёте
    rs = P["stream_radius"]
    sigma = rs * (0.6 + 0.12 * math.sqrt(v_imp / 100.0))        # ширина "пятна" удара
    r_coil = rs * 0.5 * math.sqrt(H / 20.0 + 0.05)              # укладка струи витками
    f_coil = 2.0 + H / 25.0                                     # Гц
    Tp = V / P["pour_rate"]
    dt_dep = 0.025
    nsteps = max(int(math.ceil(Tp / dt_dep)), 1)
    dvol = V / nsteps
    wob = np.zeros(2)
    phase = rng.uniform(0, 2 * math.pi)
    splash_p = min(0.3, max(0.0, (H - 80.0) / 800.0))
    n_splash = 0
    for k in range(nsteps):
        # процесс Орнштейна-Уленбека для дрожания руки
        wob += -wob * (dt_dep / 0.6) + P["pour_wobble"] * math.sqrt(2 * dt_dep / 0.6) * rng.normal(size=2)
        phase += 2 * math.pi * f_coil * dt_dep * rng.uniform(0.8, 1.2)
        px = wob[0] + r_coil * math.cos(phase)
        py = wob[1] + r_coil * math.sin(phase)
        wax.deposit(px, py, sigma * rng.uniform(0.85, 1.15), dvol, T_arr, Xc, Yc)
        if rng.random() < splash_p:
            for _ in range(rng.integers(1, 3)):
                a = rng.uniform(0, 2 * math.pi)
                dist = sigma * rng.uniform(2, 5) * (1 + H / 200.0)
                sv = min(rng.uniform(0.05, 0.6) * (1 + H / 300.0), 0.5 * dvol)
                wax.h *= 1.0 - sv / max(wax.h.sum() * (dx * f) ** 2, 1e-9)   # объём брызг - из общего
                wax.deposit(px + dist * math.cos(a), py + dist * math.sin(a),
                            max(rng.uniform(0.25, 0.5), 0.6 * dx * f), sv, T_arr - 15, Xc, Yc)
                n_splash += 1
        wax.flow(dt_dep)
    log(f"  налив: {Tp:.2f} с, удар {v_imp / 1000:.2f} м/с, брызг {n_splash}")

    # --- 2. пауза ---
    wax.flow(P["delay"])
    # перенос на точную сетку с сохранением объёма
    ic = (np.arange(n) - (n - 1) / 2) / f + (nc - 1) / 2
    I, J = np.meshgrid(ic, ic, indexing="ij")
    vol_c = wax.h.sum() * (dx * f) ** 2
    fine = Wax(n, dx, P, rng)
    fine.h = np.maximum(ndimage.map_coordinates(wax.h, [I, J], order=1), 0.0)
    fine.h[fine.h < 0.02] = 0.0
    fine.h *= vol_c / max(fine.h.sum() * dx * dx, 1e-12)
    fine.h = ndimage.gaussian_filter(fine.h, 0.6 * f)      # сгладить ступеньки грубой сетки
    fine.h[fine.h < 0.02] = 0.0
    fine.h *= vol_c / max(fine.h.sum() * dx * dx, 1e-12)
    fine.T = ndimage.map_coordinates(wax.T, [I, J], order=1)
    wax = fine
    mask = wax.h > 0.05
    m = wax.h * mask
    cx = (X * m).sum() / m.sum()
    cy = (Y * m).sum() / m.sum()
    log(f"  перед прижимом: толщина макс {wax.h.max():.2f} мм, "
        f"T центр {wax.T[wax.h > 0.5].mean() if (wax.h > 0.5).any() else 0:.0f}°C, "
        f"центр лужицы ({cx:.2f}, {cy:.2f})")

    # --- 3. прижим и выдержка ---
    cache = {}
    sx, sy = cx + P["offset_x"], cy + P["offset_y"]

    def ceil_at(z):
        return stamp_ceiling(X, Y, stamp, args.stamp_size, sx, sy, P["rotation"], P["tilt"],
                             P["tilt_dir"], not args.no_mirror, args.relief_depth, z, cache)

    rel = ceil_at(0.0)
    finite = np.isfinite(rel)
    rel_min = rel[finite].min()
    z = wax.h.max() - rel_min + 0.3
    dz = args.press_step
    min_film = args.min_film
    t_hold = None
    dwell_left = P["dwell"]
    F = 0.0
    info = dict(contact_area=0.0, film=0.0, T_contact=0.0)
    steps = 0
    while steps < 20000:
        steps += 1
        c = rel + z
        wax.redistribute(c)
        contact = finite & (wax.h >= c - 1e-4) & (wax.h > 0.01)
        wax.contact = contact
        A = contact.sum() * dx * dx
        if A > 0:
            film = float(c[contact].mean())
            R = math.sqrt(A / math.pi)
            Tc = float(wax.T[contact].mean())
            p = wax.yield_stress(Tc) * (1.0 + R / (3.0 * max(film, 0.05)))
            F = p * A * 1e-6
            if t_hold is None:
                info.update(contact_area=A, film=film, T_contact=Tc)
        else:
            F = 0.0
        bottom = (rel_min + z) <= min_film
        if F < P["force"] and not bottom:
            z -= dz
            dt = dz / P["press_speed"]
        else:
            if t_hold is None:
                t_hold = 0.0
                info["force_at_stop"] = F
                log(f"  штамп остановился: F={F:.1f} Н, площадь контакта {A:.0f} мм^2, "
                    f"плёнка {info['film']:.2f} мм, T контакта {info['T_contact']:.0f}°C"
                    + (" (упёрся в бумагу)" if bottom else ""))
            dt = 0.25
        if t_hold is not None:
            if dwell_left <= 0:
                break
            dt = min(dt, dwell_left)
            dwell_left -= dt
        wax.flow(dt, blocked=contact)
    # --- 4. подъём штампа: небольшая релаксация ещё мягкого сургуча ---
    wax.contact[:] = False
    wax.flow(args.post_flow)
    log(f"  к концу выдержки сургуч выдержал бы {F:.0f} Н (застыл); "
        f"время расчёта {time.time() - t0:.1f} с")
    info.update(stamp_center=[sx, sy], puddle_center=[cx, cy],
                max_height=float(wax.h.max()), volume_mm3=float(wax.h.sum() * dx * dx))
    return wax.h, coords, info


# ----------------------------------------------------------------------------
# Экспорт
# ----------------------------------------------------------------------------

def crop(h, margin_px):
    rows = np.flatnonzero((h > 0.01).any(axis=1))
    cols = np.flatnonzero((h > 0.01).any(axis=0))
    if len(rows) == 0:
        return h
    r0, r1 = max(rows[0] - margin_px, 0), min(rows[-1] + margin_px + 1, h.shape[0])
    c0, c1 = max(cols[0] - margin_px, 0), min(cols[-1] + margin_px + 1, h.shape[1])
    return h[r0:r1, c0:c1]


def save_heightmap(path, h, dx, zmax=None):
    zmax = zmax or float(h.max()) or 1.0
    data = np.clip(h / zmax, 0, 1) * 65535.0
    Image.fromarray(np.round(data).astype(np.uint16)).save(path)
    meta = dict(file=os.path.basename(path), mm_per_pixel=dx, z_mm_at_white=zmax,
                width_mm=h.shape[1] * dx, height_mm=h.shape[0] * dx,
                note="16 бит; высота_мм = значение/65535 * z_mm_at_white")
    return meta


def save_preview(path, h, dx):
    gy, gx = np.gradient(h, dx)
    az, alt = math.radians(315), math.radians(40)
    lx, ly, lz = math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)
    nz = 1.0 / np.sqrt(1 + gx ** 2 + gy ** 2)
    # строка изображения растёт вниз, поэтому gy берём со знаком минус
    shade = (-gx * lx + gy * ly + lz) * nz
    shade = np.clip(shade, 0, 1)
    wax = h > 0.02
    rgb = np.zeros(h.shape + (3,))
    base = np.array([0.62, 0.06, 0.08])
    rgb[wax] = (base[None, :] * (0.25 + 1.1 * shade[wax, None])).clip(0, 1)
    rgb[~wax] = np.array([0.93, 0.89, 0.80]) * (0.9 + 0.1 * shade[~wax, None])
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def save_stl(path, h, dx, base=0.0, eps=0.02):
    """Замкнутое тело: верх - поверхность сургуча, низ - плоскость z=0."""
    ny, nx = h.shape
    xs = (np.arange(nx) - (nx - 1) / 2) * dx
    ys = -(np.arange(ny) - (ny - 1) / 2) * dx
    Xg, Yg = np.meshgrid(xs, ys)
    Z = h + base
    if base > 0:
        cell = np.ones((ny - 1, nx - 1), bool)
    else:
        on = h > eps
        cell = on[:-1, :-1] & on[1:, :-1] & on[:-1, 1:] & on[1:, 1:]
    ci, cj = np.nonzero(cell)

    def P(i, j, top=True):
        return np.stack([Xg[i, j], Yg[i, j], Z[i, j] if top else np.zeros_like(Z[i, j])], -1)

    tris = []
    p00, p01, p10, p11 = (ci, cj), (ci, cj + 1), (ci + 1, cj), (ci + 1, cj + 1)
    for top in (True, False):
        a = np.stack([P(*p10, top), P(*p11, top), P(*p01, top)], 1)
        b = np.stack([P(*p10, top), P(*p01, top), P(*p00, top)], 1)
        t = np.concatenate([a, b])
        if not top:
            t = t[:, ::-1]
        tris.append(t)
    # стенки: рёбра между активной и неактивной клетками
    pad = np.pad(cell, 1)
    # горизонтальные рёбра (между строками клеток): строка точек i
    for di, sgn in ((-1, 1.0), (1, -1.0)):
        nb = pad[1 + di:pad.shape[0] - 1 + di, 1:-1]
        ei, ej = np.nonzero(cell & ~nb)
        pi = ei if di == -1 else ei + 1
        q0, q1 = (pi, ej), (pi, ej + 1)
        tris.append(_wall(P(*q0), P(*q1), P(*q0, False), P(*q1, False), np.array([0, sgn, 0])))
    for dj, sgn in ((-1, -1.0), (1, 1.0)):
        nb = pad[1:-1, 1 + dj:pad.shape[1] - 1 + dj]
        ei, ej = np.nonzero(cell & ~nb)
        pj = ej if dj == -1 else ej + 1
        q0, q1 = (ei, pj), (ei + 1, pj)
        tris.append(_wall(P(*q0), P(*q1), P(*q0, False), P(*q1, False), np.array([sgn, 0, 0])))
    tris = np.concatenate([t for t in tris if len(t)]).astype(np.float32)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = nrm
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(b"Splotch wax seal simulation".ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(rec.tobytes())
    return len(tris)


def _wall(t0, t1, b0, b1, outward):
    if len(t0) == 0:
        return np.zeros((0, 3, 3))
    tri = np.concatenate([np.stack([t0, b0, b1], 1), np.stack([t0, b1, t1], 1)])
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = (nrm @ outward) < 0
    tri[flip] = tri[flip][:, ::-1]
    return tri


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Splotch - симуляция сургучной печати: налив, растекание, прижим штампа, экспорт STL/карты высот.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = ap.add_argument_group("штамп")
    g.add_argument("--stamp", help="карта высот штампа (PNG/JPG/TIFF, 8 или 16 бит)")
    g.add_argument("--stamp-size", type=float, default=25.0, help="ширина изображения штампа, мм (25)")
    g.add_argument("--relief-depth", type=float, default=1.2,
                   help="перепад высот рельефа штампа (от 'пусто' до белого), мм (1.2)")
    g.add_argument("--empty", choices=["black", "white"], default="black",
                   help="какой цвет означает отсутствие материала (black)")
    g.add_argument("--empty-threshold", type=float, default=0.02,
                   help="допуск цвета 'пусто', доля 0..1 (0.02)")
    g.add_argument("--holes-through", action="store_true",
                   help="пустые области внутри контура - сквозные (иначе - самое глубокое углубление)")
    g.add_argument("--no-mirror", action="store_true",
                   help="не зеркалить (если карта уже нарисована как готовый оттиск)")
    g.add_argument("--stamp-blur", type=float, default=0.04,
                   help="сглаживание штампа, мм - предел передачи мелких деталей (0.04)")
    g.add_argument("--make-demo-stamp", metavar="PNG", help="создать демо-штамп и выйти")

    g = ap.add_argument_group("параметры симуляции (число, a:b или m~s)")
    for k, (d, lo, hi, desc) in SIM_PARAMS.items():
        g.add_argument("--" + k.replace("_", "-"), type=Spec, default=Spec(d), help=f"{desc} ({d})")

    g = ap.add_argument_group("численные")
    g.add_argument("--resolution", type=float, default=0.15, help="шаг сетки, мм (0.15)")
    g.add_argument("--canvas", type=float, default=None, help="размер поля, мм (авто)")
    g.add_argument("--pour-coarsen", type=int, default=2,
                   help="во сколько раз грубее сетка на этапе налива (2; 1 = как основная)")
    g.add_argument("--press-step", type=float, default=0.02, help="шаг опускания штампа, мм (0.02)")
    g.add_argument("--min-film", type=float, default=0.15,
                   help="минимальная толщина под штампом (штамп упёрся), мм (0.15)")
    g.add_argument("--post-flow", type=float, default=1.0,
                   help="время растекания после подъёма штампа, с (1)")

    g = ap.add_argument_group("вывод")
    g.add_argument("--out", default="splotch", help="префикс выходных файлов (splotch)")
    g.add_argument("--format", default="stl,png",
                   help="через запятую: stl, png (16-бит карта высот), npy (stl,png)")
    g.add_argument("--hm-zmax", type=float, default=None,
                   help="высота (мм), соответствующая белому в карте высот (авто = максимум)")
    g.add_argument("--stl-base", type=float, default=0.0,
                   help="добавить подложку этой толщины под всё поле, мм (0 = только сургуч)")
    g.add_argument("--stl-step", type=int, default=1,
                   help="прореживание сетки для STL (1 = полное разрешение)")
    g.add_argument("--no-crop", action="store_true", help="не обрезать поле по сургучу")
    g.add_argument("--no-preview", action="store_true", help="не сохранять PNG-превью")
    g.add_argument("--count", type=int, default=1, help="сколько вариантов сгенерировать (1)")
    g.add_argument("--seed", type=int, default=None, help="зерно генератора случайных чисел")
    g.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    # отрицательные значения вида "--offset-x -1.5:1.5" склеиваем в "--offset-x=-1.5:1.5"
    argv = list(sys.argv[1:] if argv is None else argv)
    fixed = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if (tok.startswith("--") and "=" not in tok and i + 1 < len(argv)
                and re.match(r"^-\.?\d", argv[i + 1])):
            fixed.append(f"{tok}={argv[i + 1]}")
            i += 2
            continue
        fixed.append(tok)
        i += 1
    args = ap.parse_args(fixed)
    if args.make_demo_stamp:
        make_demo_stamp(args.make_demo_stamp)
        return
    if not args.stamp:
        ap.error("нужен --stamp (или --make-demo-stamp для создания примера)")
    log = (lambda *a: None) if args.quiet else print
    blur_px = 0.0
    stamp_img = Image.open(args.stamp)
    blur_px = args.stamp_blur * stamp_img.width / args.stamp_size
    stamp = load_stamp(args.stamp, args.empty, args.empty_threshold, args.holes_through, blur_px)
    formats = {f.strip().lower() for f in args.format.split(",") if f.strip()}
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for k in range(args.count):
        rng = np.random.default_rng([seed, k])
        P = {}
        for name, (_, lo, hi, _) in SIM_PARAMS.items():
            P[name] = float(np.clip(getattr(args, name).sample(rng), lo, hi))
        prefix = args.out if args.count == 1 else f"{args.out}_{k + 1:03d}"
        log(f"[{k + 1}/{args.count}] {prefix}")
        log("  " + ", ".join(f"{n}={v:.3g}" for n, v in P.items()))
        h, coords, info = simulate(P, stamp, args, rng, log)
        dx = args.resolution
        if not args.no_crop:
            h = crop(h, int(2.0 / dx))
        meta = dict(seed=seed, index=k, params=P,
                    specs={n: getattr(args, n).text for n in SIM_PARAMS},
                    stamp=os.path.abspath(args.stamp), stamp_size_mm=args.stamp_size,
                    relief_depth_mm=args.relief_depth, mirror=not args.no_mirror,
                    resolution_mm=dx, result=info)
        if "png" in formats:
            meta["heightmap"] = save_heightmap(prefix + "_height.png", h, dx, args.hm_zmax)
            log(f"  -> {prefix}_height.png (белый = {meta['heightmap']['z_mm_at_white']:.3f} мм)")
        if "npy" in formats:
            np.save(prefix + "_height.npy", h.astype(np.float32))
            log(f"  -> {prefix}_height.npy (мм)")
        if "stl" in formats:
            s = max(args.stl_step, 1)
            ntri = save_stl(prefix + ".stl", h[::s, ::s], dx * s, args.stl_base)
            log(f"  -> {prefix}.stl ({ntri} треугольников)")
        if not args.no_preview:
            save_preview(prefix + "_preview.png", h, dx)
        with open(prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
