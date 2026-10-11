"""
Procedural planet generator — v8
================================
GPU/CPU hybrid generation: icosphere (+ adaptive refinement) -> terrain
(compute shader or rotated fbm CPU) -> hydraulic erosion -> thermal erosion
(GPU or CPU) -> glaciers & fjords -> rivers (deltas, lakes) -> climate
(seasons) -> HDR scene: raymarched atmosphere, volumetric or shell clouds
with cyclones + synced shadows, water plane with screen-space refraction,
god rays, sun body + corona + lens flare, instanced vegetation, rings,
companion planet, ringed moons with atmospheres, precipitation, bloom,
ACES tonemap, survey probe, PNG-embedded config sharing.

PySide6 + PyOpenGL + numpy.  Optional: imageio (GIF export).
"""

import argparse
import ctypes
import json
import math
import os
import random
import struct
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QColor, QImage, QKeySequence, QMatrix4x4, QPalette,
                           QPixmap, QShortcut, QVector2D, QVector3D, QVector4D,
                           QSurfaceFormat)
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)
from OpenGL import GL

_CFG_PATH = (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "planet_last.json")
             if "__file__" in globals() else "planet_last.json")
_QUALITY = {"Low": (12, 4), "Medium": (20, 5), "High": (30, 6)}


# ---------------------------------------------------------------------------
# Simplex noise (CPU, vectorized) + fractal helpers
# ---------------------------------------------------------------------------

_GRAD3 = np.array(
    [[ 1,  1, 0], [-1,  1, 0], [ 1, -1, 0], [-1, -1, 0],
     [ 1,  0, 1], [-1,  0, 1], [ 1,  0,-1], [-1,  0,-1],
     [ 0,  1, 1], [ 0, -1, 1], [ 0,  1,-1], [ 0, -1,-1]], dtype=np.float64)


class SimplexNoise3D:
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256).astype(np.int32)
        self.perm = np.concatenate([p, p])
        self.perm12 = (self.perm % 12).astype(np.int32)

    def __call__(self, x, y, z):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        z = np.ascontiguousarray(z, dtype=np.float64)
        s = (x + y + z) * self.F3
        i = np.floor(x + s).astype(np.int32)
        j = np.floor(y + s).astype(np.int32)
        k = np.floor(z + s).astype(np.int32)
        t = (i + j + k) * self.G3
        x0 = x - (i - t); y0 = y - (j - t); z0 = z - (k - t)
        xy, yz, xz = x0 >= y0, y0 >= z0, x0 >= z0
        a = xy & yz; b = xy & ~yz & xz; c = xy & ~yz & ~xz
        d = ~xy & ~yz; e = ~xy & yz & ~xz; f = ~xy & yz & xz
        i1 = (a | b).astype(np.int32); j1 = (e | f).astype(np.int32)
        k1 = (c | d).astype(np.int32)
        i2 = (a | b | f).astype(np.int32); j2 = (a | d | e).astype(np.int32)
        k2 = (b | c | d | e).astype(np.int32)
        x1 = x0 - i1 + self.G3; y1 = y0 - j1 + self.G3; z1 = z0 - k1 + self.G3
        x2 = x0 - i2 + 2.0 * self.G3; y2 = y0 - j2 + 2.0 * self.G3
        z2 = z0 - k2 + 2.0 * self.G3
        x3 = x0 - 1.0 + 3.0 * self.G3; y3 = y0 - 1.0 + 3.0 * self.G3
        z3 = z0 - 1.0 + 3.0 * self.G3
        P, P12 = self.perm, self.perm12
        ii, jj, kk = i & 255, j & 255, k & 255
        g0 = P12[ii + P[jj + P[kk]]]
        g1 = P12[ii + i1 + P[jj + j1 + P[kk + k1]]]
        g2 = P12[ii + i2 + P[jj + j2 + P[kk + k2]]]
        g3 = P12[ii + 1 + P[jj + 1 + P[kk + 1]]]
        n = (self._corner(x0, y0, z0, g0) + self._corner(x1, y1, z1, g1)
             + self._corner(x2, y2, z2, g2) + self._corner(x3, y3, z3, g3))
        return 32.0 * n

    @staticmethod
    def _corner(dx, dy, dz, g):
        t = np.maximum(0.6 - dx * dx - dy * dy - dz * dz, 0.0)
        grad = _GRAD3[g]
        dot = grad[:, 0] * dx + grad[:, 1] * dy + grad[:, 2] * dz
        return (t * t) * (t * t) * dot


def _rotation_matrices():
    rots = []
    for o in range(8):
        ax, ay = o * 1.7 + 0.9, o * 2.3 + 0.4
        cx, sx, cy, sy = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay)
        rots.append(np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                    @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))
    return rots


_OCTAVE_ROTS = _rotation_matrices()


def _rot_domain(R, x, y, z):
    return (R[0, 0] * x + R[0, 1] * y + R[0, 2] * z,
            R[1, 0] * x + R[1, 1] * y + R[1, 2] * z,
            R[2, 0] * x + R[2, 1] * y + R[2, 2] * z)


def fbm(noise, x, y, z, octaves=5, lacunarity=2.0, gain=0.5):
    total = np.zeros_like(x)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        total += amp * noise(x * freq, y * freq, z * freq)
        norm += amp; amp *= gain; freq *= lacunarity
        if o + 1 < octaves:
            x, y, z = _rot_domain(_OCTAVE_ROTS[o % 8], x, y, z)
    return total / norm


def ridged(noise, x, y, z, octaves=6, lacunarity=2.1, gain=0.5, power=1.8):
    total = np.zeros_like(x)
    amp, freq, norm, prev = 0.5, 1.0, 0.0, 1.0
    for o in range(octaves):
        n = 1.0 - np.abs(noise(x * freq, y * freq, z * freq))
        n = n ** power * prev
        prev = np.clip(n * 2.0, 0.0, 1.0)
        total += n * amp; norm += amp; amp *= gain; freq *= lacunarity
        if o + 1 < octaves:
            x, y, z = _rot_domain(_OCTAVE_ROTS[(o + 3) % 8], x, y, z)
    return total / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Icosphere, neighbors, adaptive refinement
# ---------------------------------------------------------------------------

_T = (1.0 + math.sqrt(5.0)) / 2.0
_ICO_VERTS = np.array([
    [-1,  _T,  0], [ 1,  _T,  0], [-1, -_T,  0], [ 1, -_T,  0],
    [ 0, -1,  _T], [ 0,  1,  _T], [ 0, -1, -_T], [ 0,  1, -_T],
    [ _T,  0, -1], [ _T,  0,  1], [-_T,  0, -1], [-_T,  0,  1],
], dtype=np.float64)
_ICO_FACES = np.array([
    [ 0, 11,  5], [ 0,  5,  1], [ 0,  1,  7], [ 0,  7, 10], [ 0, 10, 11],
    [ 1,  5,  9], [ 5, 11,  4], [11, 10,  2], [10,  7,  6], [ 7,  1,  8],
    [ 3,  9,  4], [ 3,  4,  2], [ 3,  2,  6], [ 3,  6,  8], [ 3,  8,  9],
    [ 4,  9,  5], [ 2,  4, 11], [ 6,  2, 10], [ 8,  6,  7], [ 9,  8,  1],
], dtype=np.int64)


def _subdivide(verts, faces):
    nf = len(faces)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    uniq, inv = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True)
    mids = verts[uniq[:, 0]] + verts[uniq[:, 1]]
    mids /= np.linalg.norm(mids, axis=1, keepdims=True)
    base = len(verts)
    verts = np.vstack([verts, mids])
    m = base + inv
    mab, mbc, mca = m[:nf], m[nf:2 * nf], m[2 * nf:]
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    faces = np.vstack([np.column_stack([a, mab, mca]),
                       np.column_stack([b, mbc, mab]),
                       np.column_stack([c, mca, mbc]),
                       np.column_stack([mab, mbc, mca])])
    return verts, faces


def build_icosphere(subdiv):
    verts, faces = _ICO_VERTS.copy(), _ICO_FACES.copy()
    cent = verts[faces].mean(axis=1)
    nrm = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                   verts[faces[:, 2]] - verts[faces[:, 0]])
    flip = np.einsum("ij,ij->i", nrm, cent) < 0
    faces[flip] = faces[flip, ::-1]
    for _ in range(subdiv):
        verts, faces = _subdivide(verts, faces)
    return verts, faces


_ICOSPHERE_CACHE = {}


def get_icosphere(subdiv):
    if subdiv not in _ICOSPHERE_CACHE:
        _ICOSPHERE_CACHE[subdiv] = build_icosphere(subdiv)
    return _ICOSPHERE_CACHE[subdiv]


def build_neighbors(faces, n):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                   faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]])
    order = np.argsort(e[:, 0], kind="stable")
    src = e[order, 0].astype(np.int64)
    dst = e[order, 1].astype(np.int64)
    counts = np.bincount(src, minlength=n)
    maxk = int(counts.max())
    nb = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, maxk))
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    slots = np.arange(len(dst), dtype=np.int64) - np.repeat(starts, counts)
    nb[src, slots] = dst
    return nb


def _cheap_land(dirs, p):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    terrain = SimplexNoise3D(p.seed)
    warp_n = SimplexNoise3D(p.seed + 101)
    if p.warp > 0.0:
        w = fbm(warp_n, x + 5.2, y + 1.3, z + 7.9, 2) * p.warp
        qx, qy, qz = x + w, y + w * 0.9, z + w * 1.1
    else:
        qx, qy, qz = x, y, z
    s = p.continent_scale
    return fbm(terrain, qx * s, qy * s, qz * s, 4) - p.sea_level


def _refine(verts, faces, want):
    F = len(faces)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    uniq, inv = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True)
    idx01, idx12, idx20 = inv[:F], inv[F:2 * F], inv[2 * F:]
    want_edge = np.zeros(len(uniq), bool)
    want_edge[idx01[want]] = True
    want_edge[idx12[want]] = True
    want_edge[idx20[want]] = True
    mids = (verts[uniq[:, 0]] + verts[uniq[:, 1]]) / 2.0
    mids /= np.linalg.norm(mids, axis=1, keepdims=True)
    base = len(verts)
    verts = np.vstack([verts, mids])
    mid = base + inv
    m01, m12, m20 = mid[:F], mid[F:2 * F], mid[2 * F:]
    k = (want_edge[idx01].astype(np.int8) + want_edge[idx12] + want_edge[idx20])
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    out = [faces[k == 0]]
    m3 = k == 3
    out += [np.column_stack([a[m3], m01[m3], m20[m3]]),
            np.column_stack([m01[m3], b[m3], m12[m3]]),
            np.column_stack([m20[m3], m12[m3], c[m3]]),
            np.column_stack([m01[m3], m12[m3], m20[m3]])]
    s = (k == 1) & want_edge[idx01]
    out += [np.column_stack([a[s], m01[s], c[s]]),
            np.column_stack([m01[s], b[s], c[s]])]
    s = (k == 1) & want_edge[idx12]
    out += [np.column_stack([a[s], b[s], m12[s]]),
            np.column_stack([a[s], m12[s], c[s]])]
    s = (k == 1) & want_edge[idx20]
    out += [np.column_stack([a[s], b[s], m20[s]]),
            np.column_stack([m20[s], b[s], c[s]])]
    q = (k == 2) & want_edge[idx01] & want_edge[idx12]
    out += [np.column_stack([m01[q], b[q], m12[q]]),
            np.column_stack([a[q], m01[q], m12[q]]),
            np.column_stack([a[q], m12[q], c[q]])]
    q = (k == 2) & want_edge[idx12] & want_edge[idx20]
    out += [np.column_stack([m12[q], c[q], m20[q]]),
            np.column_stack([b[q], m12[q], m20[q]]),
            np.column_stack([b[q], m20[q], a[q]])]
    q = (k == 2) & want_edge[idx20] & want_edge[idx01]
    out += [np.column_stack([m20[q], a[q], m01[q]]),
            np.column_stack([c[q], m20[q], m01[q]]),
            np.column_stack([c[q], m01[q], b[q]])]
    return verts, np.vstack(out)


def adaptive_icosphere(subdiv, rounds, p):
    verts, faces = get_icosphere(subdiv)
    if rounds <= 0 or subdiv >= 7:
        return verts, faces
    for _ in range(int(rounds)):
        if len(faces) > 420_000:
            break
        land = _cheap_land(verts, p)
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        coast = ((np.abs(land[f0]) < 0.045) | (np.abs(land[f1]) < 0.045)
                 | (np.abs(land[f2]) < 0.045))
        tri = verts[faces]
        nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        ctr = tri.mean(axis=1)
        ctr /= np.maximum(np.linalg.norm(ctr, axis=1, keepdims=True), 1e-12)
        steep = (1.0 - np.abs(np.einsum("ij,ij->i", nn, ctr))) > 0.04
        want = coast | steep
        if not want.any():
            break
        verts, faces = _refine(verts, faces, want)
    return verts, faces


# ---------------------------------------------------------------------------
# Parameters, palettes, styles
# ---------------------------------------------------------------------------

@dataclass
class PlanetParams:
    seed: int
    subdiv: int = 6
    adaptive: int = 1
    continent_scale: float = 1.5
    sea_level: float = 0.03
    mountain_height: float = 0.14
    warp: float = 0.40
    detail: float = 1.0
    impacts: int = 0
    volcanoes: int = 0
    erosion: float = 0.0
    thermal: int = 0
    rivers: float = 0.0
    rainfall: float = 0.0
    climate: float = 0.0
    season: float = 0.0          # -1..1: hemispheric insolation asymmetry
    age: float = 0.0             # Myr of extra erosion/thermal
    terraform: float = 0.0       # 0..1 mars -> terran
    style: str = "terran"


DEEP      = np.array([0.010, 0.050, 0.160])
SHALLOW   = np.array([0.060, 0.330, 0.440])
SEA_ICE   = np.array([0.820, 0.880, 0.920])
SAND      = np.array([0.760, 0.700, 0.510])
GRASS     = np.array([0.130, 0.370, 0.120])
FOREST    = np.array([0.050, 0.240, 0.090])
DESERT    = np.array([0.800, 0.680, 0.450])
TUNDRA    = np.array([0.420, 0.400, 0.320])
ROCK      = np.array([0.440, 0.410, 0.380])
SNOW      = np.array([0.940, 0.950, 0.970])
RIVER     = np.array([0.100, 0.220, 0.280])
SEDIMENT  = np.array([0.230, 0.300, 0.140])
LAKE      = np.array([0.090, 0.200, 0.160])
GLACIER   = np.array([0.860, 0.910, 0.960])
TGREEN    = np.array([0.120, 0.380, 0.140])
TGREEN_D  = np.array([0.050, 0.300, 0.100])
ICE_L     = np.array([0.900, 0.950, 1.000])
ICE_D     = np.array([0.600, 0.740, 0.860])
PALE_BLUE = np.array([0.700, 0.800, 0.900])
COLD_ROCK = np.array([0.450, 0.470, 0.520])
RUST_L    = np.array([0.800, 0.480, 0.260])
RUST_D    = np.array([0.540, 0.280, 0.140])
MARS_ROCK = np.array([0.450, 0.260, 0.160])
MARS_LOW  = np.array([0.380, 0.200, 0.120])
MARS_CAP  = np.array([0.920, 0.890, 0.860])
BASALT_D  = np.array([0.100, 0.090, 0.100])
BASALT_L  = np.array([0.240, 0.210, 0.200])
LAVA_CRUST = np.array([0.250, 0.050, 0.020])
LAVA_BRIGHT = np.array([1.000, 0.500, 0.100])
MOON_C    = np.array([0.600, 0.580, 0.550])
MARIA     = np.array([0.160, 0.160, 0.180])
CREAM     = np.array([0.870, 0.800, 0.640])
TAN_D     = np.array([0.580, 0.440, 0.300])
BROWN     = np.array([0.420, 0.280, 0.190])
STORM     = np.array([0.760, 0.350, 0.200])

STYLES = {
    "terran": dict(glow=(0, 0, 0), water_glow=(0, 0, 0), cloud=(1, 1, 1),
                   cloud_max=0.95, night=1.0, atmo=True,
                   betaR=(0.30, 0.75, 1.90), aurora=(0.15, 0.95, 0.50),
                   sky=(0.35, 0.55, 0.95), deep=(0.04, 0.13, 0.28)),
    "ice":    dict(glow=(0, 0, 0), water_glow=(0, 0, 0), cloud=(1, 1, 1),
                   cloud_max=0.85, night=0.0, atmo=True,
                   betaR=(0.35, 0.85, 1.95), aurora=(0.35, 0.75, 1.00),
                   sky=(0.60, 0.75, 0.95), deep=(0.30, 0.45, 0.60)),
    "mars":   dict(glow=(0, 0, 0), water_glow=(0, 0, 0), cloud=(0.85, 0.70, 0.50),
                   cloud_max=0.45, night=0.0, atmo=True,
                   betaR=(0.90, 0.45, 0.20), aurora=None,
                   sky=(0.90, 0.60, 0.35), deep=(0.22, 0.13, 0.09)),
    "lava":   dict(glow=(1.60, 0.45, 0.08), water_glow=(0.75, 0.20, 0.03),
                   cloud=(0.25, 0.22, 0.20), cloud_max=0.40, night=0.0,
                   atmo=True, betaR=(0.70, 0.25, 0.10), aurora=None,
                   sky=(0.50, 0.20, 0.10), deep=(0.55, 0.20, 0.04)),
    "moon":   dict(glow=(0, 0, 0), water_glow=(0, 0, 0), cloud=(1, 1, 1),
                   cloud_max=0.0, night=0.0, atmo=False,
                   betaR=(0, 0, 0), aurora=None, sky=(0, 0, 0),
                   deep=(0, 0, 0)),
    "gas":    dict(glow=(0, 0, 0), water_glow=(0, 0, 0), cloud=(1, 1, 1),
                   cloud_max=0.0, night=0.0, atmo=True,
                   betaR=(0.45, 0.65, 1.10), aurora=None,
                   sky=(0.50, 0.55, 0.80), deep=(0, 0, 0)),
}


def _crater_list(p):
    rng = np.random.default_rng(p.seed * 7 + 13)
    n = int(p.impacts)
    c = rng.normal(size=(n, 3)); c /= np.maximum(
        np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    r = np.exp(rng.uniform(np.log(0.02), np.log(0.22), n))
    dm = rng.uniform(0.7, 1.3, n)
    return c, r, dm


def _volcano_list(p):
    rng = np.random.default_rng(p.seed * 11 + 5)
    n = int(p.volcanoes)
    c = rng.normal(size=(n, 3)); c /= np.maximum(
        np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    r = rng.uniform(0.05, 0.16, n)
    h = rng.uniform(0.06, 0.16, n)
    return c, r, h


def _craters(dirs, p, count):
    c, r, dm = _crater_list(p)
    out = np.zeros(len(dirs))
    for ci, ri, di in zip(c, r, dm):
        t = np.linalg.norm(dirs - ci, axis=1) / ri
        tt = np.minimum(t, 1.0)
        out += (1.0 - tt * tt) ** 2 * ri * 0.55 * di \
             + np.exp(-((t - 1.0) / 0.22) ** 2) * ri * 0.30 * di
    return out


def _volcanoes(dirs, p, count):
    c, r, h = _volcano_list(p)
    out = np.zeros(len(dirs)); glow = np.zeros(len(dirs))
    for ci, ri, hi in zip(c, r, h):
        t = np.linalg.norm(dirs - ci, axis=1) / ri
        out += np.clip(1.0 - t, 0.0, None) ** 1.4 * hi \
             * (1.0 - 0.5 * np.exp(-(t / 0.16) ** 2))
        glow += np.exp(-(t / 0.10) ** 2) * 0.9
    return out, glow


# ---------------------------------------------------------------------------
# Terrain geometry (CPU)
# ---------------------------------------------------------------------------

def terrain_geometry(dirs, p):
    if p.style == "gas":
        z = np.zeros(len(dirs))
        return z, z.copy()
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    terrain = SimplexNoise3D(p.seed)
    warp_n = SimplexNoise3D(p.seed + 101)
    if p.warp > 0.0:
        qx = x + p.warp * fbm(warp_n, x + 5.2, y + 1.3, z + 7.9, 3)
        qy = y + p.warp * fbm(warp_n, x + 9.7, y + 3.1, z + 2.8, 3)
        qz = z + p.warp * fbm(warp_n, x + 4.1, y + 8.6, z + 6.3, 3)
    else:
        qx, qy, qz = x, y, z
    s = p.continent_scale
    land = fbm(terrain, qx * s, qy * s, qz * s, 6) - p.sea_level
    region = fbm(terrain, qx * 1.3 + 7.7, qy * 1.3 + 3.9, qz * 1.3 + 5.2, 3)
    mask = smoothstep(0.02, 0.22, land) * smoothstep(-0.05, 0.35, region)
    mountains = mask * ridged(terrain, qx * 2.6 + 11.7, qy * 2.6 + 5.1,
                              qz * 2.6 + 9.4, 6) * p.mountain_height
    landness = smoothstep(0.0, 0.12, land)
    hills = fbm(terrain, qx * 7.0 + 21.3, qy * 7.0 + 13.7, qz * 7.0 + 31.9, 4)
    elev = land * 0.22 + mountains + hills * (0.012 + 0.055 * landness) * p.detail
    deep = smoothstep(0.03, 0.16, -land)
    spine = ridged(terrain, qx * 1.05 + 31.7, qy * 1.05 + 17.3,
                   qz * 1.05 + 23.9, 4, power=1.5)
    elev = np.where(land < 0.0, np.minimum(elev + spine * deep * 0.05, -0.003),
                    elev)
    emis = np.zeros_like(elev)
    if p.impacts > 0:
        elev = elev + _craters(dirs, p, int(p.impacts))
    if p.volcanoes > 0:
        vadd, gadd = _volcanoes(dirs, p, int(p.volcanoes))
        elev = elev + vadd
        if p.style == "lava":
            emis = emis + gadd
    return elev, emis


# ---------------------------------------------------------------------------
# Erosion (CPU), climate, glaciers, rivers
# ---------------------------------------------------------------------------

def hydraulic_erode(elev, nb, seed, passes, steps=22):
    n = len(elev)
    rng = np.random.default_rng(seed * 31 + 7)
    land = np.flatnonzero(elev > 1e-4)
    if land.size == 0 or passes <= 0:
        return elev
    M = int(min(max(2048, n // 6), 20000))
    ar = np.arange(M)
    KE, KC, KD = 0.10, 5.0, 0.10
    pos = rng.choice(land, M)
    sed = np.zeros(M)
    for _ in range(passes):
        for _ in range(steps):
            nbh = nb[pos]
            hn = elev[nbh]
            k = np.argmin(hn, axis=1)
            hlow = hn[ar, k]
            nlow = nbh[ar, k]
            slope = elev[pos] - hlow
            C = slope * KC
            dE = np.where(C > sed,
                          -np.minimum((C - sed) * KE, np.maximum(elev[pos], 0.0) * 0.4),
                          (sed - C) * KD)
            np.add.at(elev, pos, dE)
            sed = np.maximum(sed - dE, 0.0)
            pos = np.where(slope > 0, nlow, pos)
            wet = elev[pos] <= 0.0
            if wet.any():
                pos = np.where(wet, rng.choice(land, M), pos)
                sed = np.where(wet, 0.0, sed)
    return elev


def thermal_erode(elev, nb, iters, talus=0.006, rate=0.55):
    if iters <= 0 or nb is None:
        return elev
    for _ in range(iters):
        hn = elev[nb]
        f = np.clip((elev - hn.min(axis=1) - talus) * 8.0, 0.0, 1.0) * rate
        elev = elev + f * (hn.mean(axis=1) - elev)
    return elev


def climate_fields(dirs, elev, p):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    jit_n = SimplexNoise3D(p.seed + 303)
    moist_n = SimplexNoise3D(p.seed + 202)
    jitter = fbm(jit_n, x * 23.0, y * 23.0, z * 23.0, 2)
    temp = np.clip(1.0 - np.abs(dirs[:, 1]) ** 1.4 - np.maximum(elev, 0.0) * 1.7
                   + p.climate + p.season * 0.30 * dirs[:, 1]
                   + 0.06 * jitter, 0.0, 1.0)
    moist = np.clip(0.5 + 0.5 * fbm(moist_n, x * 3.1 + 4.2, y * 3.1 + 8.8,
                                    z * 3.1 + 1.9, 4) + p.rainfall, 0.0, 1.0)
    return temp, moist, jitter


def flow_accum(elev, nb):
    n = len(elev)
    ar = np.arange(n)
    hn = elev[nb]
    k = np.argmin(hn, axis=1)
    down = np.where(hn[ar, k] < elev, nb[ar, k], ar)
    order = np.argsort(-elev, kind="stable")
    acc = np.ones(n)
    for i in order:
        d = down[i]
        if d != i:
            acc[d] += acc[i]
    return down, acc


def glacier_process(elev, nb, gmask, riv):
    g = gmask.astype(np.float64)
    g = 0.5 * g + 0.5 * g[nb].mean(axis=1)
    g = 0.5 * g + 0.5 * g[nb].mean(axis=1)
    for _ in range(5):
        elev = elev + (g * 0.35) * (elev[nb].mean(axis=1) - elev)
    elev = elev - riv * g * 0.018
    fj = g * smoothstep(0.30, 0.70, riv) * smoothstep(0.055, 0.002, elev)
    return np.maximum(elev - fj * 0.09, -0.06)


def compute_rivers(elev, nb, strength):
    n = len(elev)
    down, acc = flow_accum(elev, nb)
    sea = elev <= 0.0
    rn = np.log1p(acc) / max(np.log1p(acc.max()), 1e-9)
    rn = np.maximum(rn, np.max(rn[nb], axis=1) * 0.85)
    river = np.where(elev > 0.0, smoothstep(0.50, 0.85, rn) * strength, 0.0)
    delta = river * smoothstep(0.035, 0.004, elev)
    elev = elev + delta * 0.010
    sink = np.arange(n)
    order = np.argsort(-elev, kind="stable")
    for i in order:
        d = down[i]
        sink[i] = i if d == i else sink[d]
    sink_elev = elev[np.minimum(sink, n - 1)]
    fill = 0.002 + 0.010 * rn
    lake = (~sea) & (elev < sink_elev + fill) & (acc >= 4)
    elev = np.where(lake, np.minimum(elev, sink_elev + 0.003), elev)
    river = np.where(lake, 0.0, river)
    elev = elev - river * 0.012 * smoothstep(0.004, 0.02, elev)
    return elev, river, delta, lake


# ---------------------------------------------------------------------------
# Biomes + colorize
# ---------------------------------------------------------------------------

BIOME_NAMES = (
    "Deep ocean", "Ocean", "Shallow sea", "Lake", "River", "River delta",
    "Beach", "Ice sheet", "Sea ice", "Glacier", "Snowfield", "Tundra",
    "Grassland", "Forest", "Rainforest", "Desert", "Savanna", "Highlands",
    "Mountains", "Peaks", "Volcanic highlands", "Lava sea", "Basalt plain",
    "Dust plains", "Canyon network", "Polar cap", "Maria",
    "Cratered terrain", "Cloud bands", "Storm belt")


def classify_biomes(style, water, lake, elev, h, temp, moist, river, delta,
                    glacier):
    n = len(elev)
    b = np.zeros(n, np.int8)
    if style == "terran":
        w = water & ~lake
        b[w] = np.where(elev[w] < -0.12, 0, np.where(elev[w] < -0.02, 1, 2))
        b[water & (temp < 0.08)] = 8
        b[lake] = 3
        land = ~water
        b[land & glacier] = 9
        b[land & (temp < 0.10) & ~glacier] = 10
        b[land & (h > 0.16)] = 19
        b[land & (h > 0.09) & (h <= 0.16)] = 18
        b[land & (h > 0.05) & (h <= 0.09)] = 17
        b[land & (river > 0.25)] = 4
        b[land & (delta > 0.18)] = 5
        b[land & (h < 0.006) & (temp > 0.18) & (river <= 0.25)] = 6
        b[land & (temp < 0.16) & (h <= 0.05)] = 11
        b[land & (moist < 0.30) & (h <= 0.05) & (temp >= 0.16)] = 15
        b[land & (moist > 0.68) & (temp > 0.55) & (h <= 0.05)] = 14
        b[land & (moist > 0.55) & (temp > 0.16) & (h <= 0.05)
          & (moist <= 0.68)] = 13
        b[land & (temp > 0.75) & (moist < 0.45) & (h <= 0.05)] = 16
        b[land & (b == 0)] = 12
    elif style == "ice":
        b[water & ~lake] = 1
        b[lake] = 3
        land = ~water
        b[land & glacier] = 7
        b[land & (h > 0.09) & ~glacier] = 18
        b[land & ~glacier & (h <= 0.09)] = 10
    elif style == "mars":
        b[water] = 23
        land = ~water
        b[land & (temp < 0.15)] = 25
        b[land & (h > 0.10)] = 18
        b[land & (river > 0.20)] = 24
        b[land & (b == 0)] = 23
    elif style == "lava":
        b[water] = 21
        land = ~water
        b[land & (h > 0.12)] = 20
        b[land & (b == 0)] = 22
    elif style == "moon":
        b[elev < -0.03] = 26
        b[elev >= -0.03] = 27
    elif style == "gas":
        b[:] = 28
    return b


_gas_seed = [1]


def _colorize(style, dirs, water, lake, elev, h, temp, moist, jitter, emis,
              river, delta, glacier, tf, city_n, crack_n):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    n = len(dirs)
    depth = np.clip(-elev / 0.25, 0.0, 1.0)[:, None]
    gmask = glacier[:, None].astype(np.float64) if glacier is not None else 0.0
    rough = np.zeros(n)

    if style == "terran":
        moist = np.clip(moist + 0.55 * river, 0.0, 1.0)
        ocean = mix(SHALLOW, DEEP, depth)
        ocean = mix(ocean, SEA_ICE, smoothstep(0.16, 0.08, temp)[:, None])
        ocean = np.where(lake[:, None], mix(ocean, LAKE, 0.65), ocean)
        rock = smoothstep(0.055, 0.13, h + 0.02 * jitter)
        snow = smoothstep(0.22, 0.12, temp)
        base = mix(DESERT, GRASS, smoothstep(0.30, 0.55, moist)[:, None])
        base = mix(base, FOREST, smoothstep(0.60, 0.85, moist)[:, None])
        tundra = mix(TUNDRA, SNOW, smoothstep(0.10, 0.03, temp)[:, None])
        base = mix(tundra, base, smoothstep(0.12, 0.26, temp)[:, None])
        base = mix(base, ROCK, rock[:, None])
        beach = smoothstep(0.012, 0.002, h) * smoothstep(0.18, 0.30, temp)
        base = mix(base, SAND, beach[:, None])
        base = mix(base, RIVER, np.clip(river * 0.75, 0.0, 1.0)[:, None])
        base = mix(base, SEDIMENT, np.clip(delta * 2.2, 0.0, 1.0)[:, None])
        base = mix(base, SNOW, snow[:, None])
        base = mix(base, GLACIER, 0.85 * gmask)
        cities = (smoothstep(0.30, 0.55, moist) * smoothstep(0.16, 0.30, temp)
                  * (1.0 - rock) * (1.0 - snow) * (1.0 - glacier)
                  * smoothstep(0.10, 0.02, h)
                  * smoothstep(0.35, 0.70,
                               0.5 + 0.5 * fbm(city_n, x * 9.1, y * 9.1,
                                               z * 9.1, 4)))
        emis = np.where(water, 0.0, emis + 1.3 * cities)
        water_col = ocean
        rough = 0.35 + 0.65 * rock
        rough = mix(rough, 0.55, smoothstep(0.60, 0.85, moist))
        rough = mix(rough, 0.70, smoothstep(0.30, 0.10, moist)
                    * smoothstep(0.20, 0.50, temp) * smoothstep(0.06, 0.01, h))
        rough = mix(rough, 0.60, beach)
        rough = mix(rough, 0.08, snow)
        rough = mix(rough, 0.12, glacier.astype(float))
        rough = np.where(water, 0.0, rough)

    elif style == "ice":
        ocean = mix(ICE_D, ICE_L, np.clip(0.5 + 0.6 * jitter, 0.0, 1.0)[:, None])
        base = mix(SNOW, PALE_BLUE, (0.6 * smoothstep(0.25, 0.80, moist))[:, None])
        rk = smoothstep(0.06, 0.14, h + 0.02 * jitter)
        base = mix(base, COLD_ROCK, rk[:, None])
        base = mix(base, PALE_BLUE, (0.45 * river)[:, None])
        base = mix(base, GLACIER, 0.6 * gmask)
        water_col = ocean
        rough = mix(0.15, 0.90, rk)
        rough = mix(rough, 0.10, glacier.astype(float))
        rough = np.where(water, 0.0, rough)

    elif style == "mars":
        base = mix(RUST_D, RUST_L,
                   np.clip(0.30 + 0.55 * moist + 0.15 * jitter, 0.0, 1.0)[:, None])
        base = mix(base, MARS_ROCK,
                   smoothstep(0.06, 0.15, h + 0.02 * jitter)[:, None])
        base = mix(base, MARS_LOW, (0.85 * smoothstep(-0.01, -0.12, elev))[:, None])
        base = mix(base, MARS_LOW * 0.55, (0.8 * river)[:, None])
        cap = smoothstep(0.20, 0.10, temp)
        base = mix(base, MARS_CAP, cap[:, None])
        water_col = mix(MARS_LOW * 0.7, RUST_D, depth)
        if tf > 0.0:
            veg = (smoothstep(0.25, 0.55, moist)
                   * smoothstep(0.15, 0.35, temp) * tf)[:, None]
            green = mix(GRASS, FOREST, smoothstep(0.45, 0.80, moist)[:, None])
            base = mix(base, green, veg)
            base = mix(base, SNOW, (0.5 * tf * smoothstep(0.18, 0.10, temp))[:, None])
            water_col = mix(water_col, mix(SHALLOW, DEEP, depth), tf * 0.85)
        rough = np.full(n, 0.55)
        rough = mix(rough, 0.15, cap)
        rough = mix(rough, 0.95, smoothstep(0.20, 0.40, river))
        rough = np.where(water, 0.0, rough)

    elif style == "lava":
        base = mix(BASALT_D, BASALT_L,
                   np.clip(0.5 + 0.5 * fbm(crack_n, x * 5.1, y * 5.1, z * 5.1, 4),
                           0.0, 1.0)[:, None])
        cr = ridged(crack_n, x * 4.3 + 3.7, y * 4.3 + 9.1, z * 4.3 + 1.3, 4,
                    power=1.3)
        emis += np.where(water, 0.0, smoothstep(0.80, 0.96, cr)) * \
            (0.5 + 0.5 * smoothstep(0.0, 0.10, h))
        base = mix(base, BASALT_D, (0.5 * smoothstep(0.10, 0.20, h))[:, None])
        base = mix(base, base * 0.4, (0.5 * river)[:, None])
        water_col = mix(LAVA_CRUST, LAVA_BRIGHT, depth)
        rough = np.where(emis > 0.2, 0.30, 0.60)
        rough = np.where(water, 0.0, rough)

    elif style == "gas":
        turb = fbm(crack_n, x * 1.3, y * 1.3, z * 1.3, 4)
        band = fbm(crack_n, x * 0.55 + 11.0, y * 6.5 + 2.4 * turb,
                   z * 0.55 + 7.0, 5)
        b = 0.5 + 0.5 * band
        base = mix(TAN_D, CREAM, smoothstep(0.25, 0.75, b)[:, None])
        base = mix(base, BROWN, (0.6 * smoothstep(0.72, 0.95, b))[:, None])
        base = mix(base, BROWN, smoothstep(0.55, 0.90, np.abs(y))[:, None])
        rng = np.random.default_rng(_gas_seed[0] * 17 + 3)
        for r0, strg in ((0.13, 0.95), (0.07, 0.70), (0.05, 0.60)):
            c = rng.normal(size=3); c[1] *= 0.35
            c /= np.linalg.norm(c)
            ri = r0 * rng.uniform(0.8, 1.3)
            d = np.linalg.norm(dirs - c, axis=1) / ri
            base = mix(base, STORM, (strg * np.exp(-d * d))[:, None])
            base = mix(base, CREAM,
                       (strg * np.exp(-((d - 1.15) / 0.35) ** 2))[:, None])
        water_col = base

    elif style == "moon":
        base = MOON_C * (0.55 + 0.55 * moist)[:, None]
        maria = smoothstep(0.02, -0.06, elev)
        base = mix(base, MARIA, (0.85 * maria)[:, None])
        base = mix(base, base * 0.4, (0.5 * river)[:, None])
        water_col = np.tile(MARIA * 0.9, (n, 1))
        rough = mix(0.80, 0.30, maria)
        rough = np.where(water, 0.0, rough)
    else:
        base = np.tile(ROCK, (n, 1))
        water_col = np.tile(DEEP, (n, 1))

    col = np.where(water[:, None], water_col, base)
    return col, emis, rough


# ---------------------------------------------------------------------------
# Naming, normals, vegetation, full generation
# ---------------------------------------------------------------------------

_NAME_A = ["Ka", "Ve", "Tha", "Ze", "Ny", "Or", "Al", "Be", "Cra", "Dra",
           "El", "Fen", "Gor", "Hu", "Ix", "Jo", "Kel", "Lu", "Ma", "Nar",
           "Oph", "Pra", "Qui", "Rho", "Sa", "Ta", "Ume", "Vor", "Wex",
           "Xa", "Yri", "Zor"]
_NAME_B = ["ra", "en", "il", "os", "ua", "eth", "ar", "ion", "ea", "ur",
           "ith", "ora", "ys", "um"]
_NAME_C = ["", "n", "s", "x", "th", "r", "m", "ss"]
_NAME_T = ["Prime", "Minor", "II", "III", "IV", "V", "VI", "VII", "IX",
           "XI", "Beta", "Gamma", "Delta"]


def planet_name(seed):
    rng = np.random.default_rng(seed * 977 + 41)
    core = rng.choice(_NAME_A) + rng.choice(_NAME_B) + rng.choice(_NAME_C)
    if rng.random() < 0.55:
        return f"{core} {rng.choice(_NAME_T)}"
    return core


def _smooth_normals(pos, faces):
    fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]],
                  pos[faces[:, 2]] - pos[faces[:, 0]])
    flat = faces.ravel()
    fnr = np.repeat(fn, 3, axis=0)
    n = np.column_stack([
        np.bincount(flat, weights=fnr[:, 0], minlength=len(pos)),
        np.bincount(flat, weights=fnr[:, 1], minlength=len(pos)),
        np.bincount(flat, weights=fnr[:, 2], minlength=len(pos)),
    ])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _scatter_trees(p, verts, positions, biome, moist):
    if p.style != "terran":
        return None
    tree_n = SimplexNoise3D(p.seed + 707)
    vx, vy, vz = verts[:, 0], verts[:, 1], verts[:, 2]
    clump = 0.5 + 0.5 * fbm(tree_n, vx * 2.2 + 3.3, vy * 2.2 + 7.7,
                            vz * 2.2 + 1.1, 3)
    dens = smoothstep(0.30, 0.62, clump)
    dens = np.where((biome == 13) | (biome == 14), dens,
                    np.where(biome == 12, dens * 0.35, 0.0))
    rng = np.random.default_rng(p.seed * 57 + 3)
    cand = np.flatnonzero(dens > 0.02)
    if cand.size == 0:
        return None
    keep = cand[rng.random(cand.size) < dens[cand]]
    if keep.size > 24000:
        keep = rng.choice(keep, 24000, replace=False)
    if keep.size == 0:
        return None
    N = keep.size
    rot = rng.random(N) * 6.28318
    sca = (0.0038 + 0.0038 * rng.random(N)) * (0.7 + 0.7 * moist[keep])
    col = np.where((biome[keep] == 14)[:, None], TGREEN_D, TGREEN)
    col = col * (0.8 + 0.5 * rng.random(N))[:, None]
    return np.column_stack([positions[keep], rot, sca, col]).astype(np.float32)


def generate_planet(p, verts=None, faces=None, scale=1.0, elev0=None, emis0=None):
    if verts is None:
        if p.style != "gas" and p.adaptive > 0 and p.subdiv < 7:
            verts, faces = adaptive_icosphere(p.subdiv, int(p.adaptive), p)
        else:
            verts, faces = get_icosphere(p.subdiv)

    if elev0 is not None:
        elev = np.asarray(elev0, np.float64).copy()
        emis = np.zeros(len(elev)) if emis0 is None else np.asarray(emis0, np.float64).copy()
    else:
        elev, emis = terrain_geometry(verts, p)

    n = len(elev)
    rocky = p.style != "gas"
    river = np.zeros(n); delta = np.zeros(n)
    lake = np.zeros(n, bool); glacier = np.zeros(n, bool)

    # geological time: age adds erosion budget on top of the sliders
    eff_ero = int(round(p.erosion + p.age * 0.10)) if rocky else 0
    eff_th = int(round(p.thermal + p.age * 0.08)) if rocky else 0

    nb = build_neighbors(faces, n) if rocky and (eff_ero > 0 or eff_th > 0
                                                 or p.rivers > 0
                                                 or p.style in ("terran", "ice")) else None
    if rocky and eff_ero > 0:
        elev = hydraulic_erode(elev, nb, p.seed, eff_ero)
    if rocky and eff_th > 0 and nb is not None:
        elev = thermal_erode(elev, nb, eff_th)

    tf = p.terraform if p.style == "mars" else 0.0
    p_eff = p
    if tf > 0.0:
        p_eff = PlanetParams(**{**asdict(p),
                                "sea_level": p.sea_level + (0.12 - p.sea_level) * tf,
                                "climate": p.climate + 0.35 * tf,
                                "rainfall": p.rainfall + 0.45 * tf})

    temp, moist, jitter = climate_fields(verts, elev, p_eff)

    if rocky and p.style in ("terran", "ice"):
        gm = smoothstep(0.30, 0.18, temp) * smoothstep(0.0, 0.02, elev)
        glacier = gm > 0.35
        if glacier.any() and nb is not None:
            _, acc0 = flow_accum(elev, nb)
            rn0 = np.log1p(acc0) / max(np.log1p(acc0.max()), 1e-9)
            elev = glacier_process(elev, nb, glacier, smoothstep(0.50, 0.85, rn0))

    if rocky and p.rivers > 0 and nb is not None:
        elev, river, delta, lake = compute_rivers(elev, nb, p.rivers)

    sea = elev < 0.0
    water = sea | lake
    # real bathymetry: sea floor keeps its depth (the water plane covers it);
    # lakes clamp flat just above the water plane so they stay visible
    radius = np.where(lake, 1.0, 1.0 + np.maximum(elev, elev))  # land+sea real
    radius = np.where(lake, 1.0, radius)
    positions = verts * radius[:, None] * scale

    ao = np.ones(n)
    if rocky and nb is not None:
        occ = np.maximum(elev[nb] - elev[:, None], 0.0).mean(axis=1)
        ao = np.clip(1.0 - occ * 12.0, 0.25, 1.0)
        ao[water] = 1.0

    _gas_seed[0] = p.seed
    city_n = SimplexNoise3D(p.seed + 404)
    crack_n = SimplexNoise3D(p.seed + 505)
    col, emis, rough = _colorize(p.style, verts, water, lake, elev,
                                 np.maximum(elev, 0.0), temp, moist, jitter,
                                 emis, river, delta, glacier, tf,
                                 city_n, crack_n)
    biome = classify_biomes(p.style, water, lake, elev, np.maximum(elev, 0.0),
                            temp, moist, river, delta, glacier)
    trees = _scatter_trees(p, verts, positions, biome, moist)
    nrm = _smooth_normals(positions, faces)

    return dict(
        positions=positions.astype(np.float32),
        normals=nrm.astype(np.float32),
        colors=col.astype(np.float32),
        water=water.astype(np.float32).reshape(-1, 1),
        emis=emis.astype(np.float32).reshape(-1, 1),
        h=elev.astype(np.float32).reshape(-1, 1),
        ao=ao.astype(np.float32).reshape(-1, 1),
        rough=rough.astype(np.float32).reshape(-1, 1),
        trees=trees,
        temp=temp.astype(np.float32), moist=moist.astype(np.float32),
        biome=biome, lake=lake, glacier=glacier,
        indices=np.ascontiguousarray(faces.astype(np.uint32)),
        n_vertices=int(n), n_faces=int(len(faces)),
        style=p.style, seed=p.seed, name=planet_name(p.seed),
        stats=dict(ocean=float(sea.mean()), lakes=float(lake.mean()),
                   glaciers=float(glacier.mean()),
                   peak=float(max(elev.max(), 0.0)),
                   rivers=float((river > 0.12).mean()),
                   trees=0 if trees is None else int(len(trees)),
                   sea=bool(sea.any())),
        cloud_radius=float(1.0 + max(elev.max(), 0.0) + 0.045),
    )


# ---------------------------------------------------------------------------
# Minimap / PNG helpers / equirect / exports
# ---------------------------------------------------------------------------

def quick_map(data, w=360, h=180):
    pos = data["positions"].astype(np.float64)
    dirs = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
    lon = np.arctan2(dirs[:, 2], dirs[:, 0])
    lat = np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0))
    px = ((lon + np.pi) / (2.0 * np.pi) * w).astype(np.int64) % w
    py = np.clip(((np.pi / 2.0 - lat) / np.pi * h).astype(np.int64), 0, h - 1)
    col = np.clip(data["colors"], 0.0, 1.0)
    canvas = np.zeros((h, w, 3)); cov = np.zeros((h, w))
    np.add.at(canvas, (py, px), col); np.add.at(cov, (py, px), 1.0)
    f = data["indices"]
    cd = dirs[f].mean(axis=1)
    cd /= np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), 1e-12)
    cpx = ((np.arctan2(cd[:, 2], cd[:, 0]) + np.pi) / (2.0 * np.pi) * w).astype(np.int64) % w
    cpy = np.clip(((np.pi / 2.0 - np.arcsin(np.clip(cd[:, 1], -1, 1))) / np.pi * h).astype(np.int64), 0, h - 1)
    np.add.at(canvas, (cpy, cpx), col[f].mean(axis=1))
    np.add.at(cov, (cpy, cpx), 1.0)
    img = canvas / np.maximum(cov, 1e-6)[:, :, None]
    holes = cov < 0.5
    for _ in range(6):
        s = (np.roll(img, 1, 0) + np.roll(img, -1, 0)
             + np.roll(img, 1, 1) + np.roll(img, -1, 1)) * 0.25
        img[holes] = s[holes]
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = arr.tobytes()
    return QImage(buf, w, h, 3 * w, QImage.Format_RGB888).copy()


def _png_chunk(tag, payload):
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def png_bytes_rgb(arr):
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(raw, 6))
            + _png_chunk(b"IEND", b""))


def write_png16(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.uint16))
    h, w = arr.shape
    raw = b"".join(b"\x00" + arr[y].astype(">u2").tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
                 + _png_chunk(b"IDAT", zlib.compress(raw, 6))
                 + _png_chunk(b"IEND", b""))


def insert_png_text(path, key, value):
    """Insert a tEXt chunk right after IHDR."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    ln = struct.unpack(">I", data[8:12])[0]
    pos = 8 + 8 + ln + 4                       # end of IHDR chunk
    payload = key.encode("latin-1") + b"\x00" + value.encode("utf-8")
    chunk = _png_chunk(b"tEXt", payload)
    with open(path, "wb") as fh:
        fh.write(data[:pos] + chunk + data[pos:])
    return True


def read_png_text(path, key):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    off = 8
    while off + 8 <= len(data):
        ln = struct.unpack(">I", data[off:off + 4])[0]
        typ = data[off + 4:off + 8]
        payload = data[off + 8:off + 8 + ln]
        if typ == b"tEXt":
            k, _, v = payload.partition(b"\x00")
            if k.decode("latin-1", "replace") == key:
                return v.decode("utf-8", "replace")
        if typ == b"IEND":
            break
        off += 12 + ln
    return None


def render_equirect_from_mesh(data, w=2048, h=1024):
    pos = data["positions"].astype(np.float64)
    dirs = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    lon = np.arctan2(dirs[:, 2], dirs[:, 0])
    lat = np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0))
    px = np.clip((lon + np.pi) / (2.0 * np.pi) * w, 0, w - 1e-6)
    py = np.clip((np.pi / 2.0 - lat) / np.pi * h, 0, h - 1e-6)
    col = data["colors"].astype(np.float64)
    hgt = data["h"].astype(np.float64).ravel()
    wat = data["water"].astype(np.float64).ravel()
    emi = data["emis"].astype(np.float64).ravel()
    canvas = np.zeros((h, w, 3))
    hcan = np.zeros((h, w)); wcan = np.zeros((h, w))
    ecan = np.zeros((h, w)); cov = np.zeros((h, w))
    for f in data["indices"]:
        x, y = px[f], py[f]
        if x.max() - x.min() > w / 2:
            x = np.where(x < w / 2, x + w, x)
        x0 = max(int(np.floor(x.min())), 0)
        x1 = min(int(np.ceil(x.max())) + 1, int(w * 1.5))
        y0 = max(int(np.floor(y.min())) - 1, 0)
        y1 = min(int(np.ceil(y.max())) + 1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        ax, ay, bx, by, cx, cy = x[0], y[0], x[1], y[1], x[2], y[2]
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-9:
            continue
        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / d
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / d
        w2 = 1.0 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any():
            continue
        rr, cc = gy[m], gx[m] % w
        ww0, ww1, ww2 = w0[m], w1[m], w2[m]
        np.add.at(canvas, (rr, cc),
                  ww0[:, None] * col[f[0]] + ww1[:, None] * col[f[1]]
                  + ww2[:, None] * col[f[2]])
        np.add.at(hcan, (rr, cc), ww0 * hgt[f[0]] + ww1 * hgt[f[1]] + ww2 * hgt[f[2]])
        np.add.at(wcan, (rr, cc), ww0 * wat[f[0]] + ww1 * wat[f[1]] + ww2 * wat[f[2]])
        np.add.at(ecan, (rr, cc), ww0 * emi[f[0]] + ww1 * emi[f[1]] + ww2 * emi[f[2]])
        np.add.at(cov, (rr, cc), 1.0)
    cov = np.maximum(cov, 1e-6)
    alb = canvas / cov[..., None]
    hgt_m = hcan / cov; wat_m = wcan / cov; emi_m = ecan / cov
    E = np.where(wat_m > 0.5, 0.0, hgt_m)
    gy, gx = np.gradient(E)
    lat_row = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    gx /= np.maximum(np.cos(lat_row), 0.15)[:, None]
    nrm = np.dstack([-gx * 70.0, -gy * 70.0, np.ones_like(E)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    L = np.array([-0.5, 0.5, 0.72]); L /= np.linalg.norm(L)
    lam = np.clip(nrm @ L, 0.0, 1.0)
    shade = np.where(wat_m > 0.5, 0.55 + 0.45 * lam, 0.30 + 0.70 * lam)
    alb = np.clip(alb * shade[..., None]
                  + emi_m[..., None] * np.array([1.0, 0.45, 0.10]) * 0.8,
                  0.0, 1.0)
    En = (E - E.min()) / max(E.max() - E.min(), 1e-6)
    return ((alb * 255.0).astype(np.uint8),
            ((nrm * 0.5 + 0.5) * 255.0).astype(np.uint8),
            (En * 255.0).astype(np.uint8),
            (En * 65535.0).astype(np.uint16))


def export_obj(path, data):
    v, n, c, f = data["positions"], data["normals"], data["colors"], data["indices"] + 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# procedural planet — vertex colors as 'v x y z r g b'\n")
        fh.write(f"# {len(v)} vertices, {len(f)} faces (includes bathymetry)\n")
        for i in range(len(v)):
            fh.write(f"v {v[i,0]:.5f} {v[i,1]:.5f} {v[i,2]:.5f} "
                     f"{c[i,0]:.4f} {c[i,1]:.4f} {c[i,2]:.4f}\n")
        for i in range(len(n)):
            fh.write(f"vn {n[i,0]:.4f} {n[i,1]:.4f} {n[i,2]:.4f}\n")
        fh.write("\n".join(f"f {a}//{a} {b}//{b} {cc}//{cc}" for a, b, cc in f))
        fh.write("\n")


def _gltf_vertices(data):
    pos = data["positions"].astype(np.float64)
    d = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    u = (np.arctan2(d[:, 2], d[:, 0]) + np.pi) / (2.0 * np.pi)
    v = (np.pi / 2.0 - np.arcsin(np.clip(d[:, 1], -1.0, 1.0))) / np.pi
    nrm = data["normals"].astype(np.float64)
    faces = data["indices"].astype(np.int64)
    fu = u[faces]
    wrap = (fu.max(axis=1) - fu.min(axis=1)) > 0.5
    dup_map = {}; dup_pos, dup_nrm, dup_uv = [], [], []
    idx = faces.copy()
    for fi in np.flatnonzero(wrap):
        for ci in range(3):
            vi = int(faces[fi, ci])
            if u[vi] < 0.5:
                if vi not in dup_map:
                    dup_map[vi] = len(pos) + len(dup_pos)
                    dup_pos.append(pos[vi]); dup_nrm.append(nrm[vi])
                    dup_uv.append((u[vi] + 1.0, v[vi]))
                idx[fi, ci] = dup_map[vi]
    pos_all = np.vstack([pos] + dup_pos) if dup_pos else pos
    nrm_all = np.vstack([nrm] + dup_nrm) if dup_nrm else nrm
    uv_all = np.column_stack([u, v])
    if dup_uv:
        uv_all = np.vstack([uv_all, np.array(dup_uv)])
    return (pos_all.astype("<f4"), nrm_all.astype("<f4"), uv_all.astype("<f4"),
            np.ascontiguousarray(idx.ravel().astype("<u4")))


def export_glb(path, data, albedo_png, normal_png):
    pos, nrm, uv, idx = _gltf_vertices(data)
    n = len(pos)
    blob = bytearray(); views = []

    def add_view(b, target):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(b)
        views.append(dict(buffer=0, byteOffset=off, byteLength=len(b), target=target))
        return len(views) - 1

    v_pos = add_view(pos.tobytes(), 34962)
    v_nrm = add_view(nrm.tobytes(), 34962)
    v_uv = add_view(uv.tobytes(), 34962)
    v_idx = add_view(idx.tobytes(), 34963)
    while len(blob) % 4:
        blob.append(0)
    v_img0 = add_view(albedo_png, 0)
    v_img1 = add_view(normal_png, 0)
    gltf = {
        "asset": {"version": "2.0", "generator": "procedural-planet-v8"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": data.get("name", "planet")}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0}]}],
        "accessors": [
            {"bufferView": v_pos, "componentType": 5126, "count": n,
             "type": "VEC3",
             "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()},
            {"bufferView": v_nrm, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": v_uv, "componentType": 5126, "count": n, "type": "VEC2"},
            {"bufferView": v_idx, "componentType": 5125, "count": int(idx.size),
             "type": "SCALAR"}],
        "materials": [{
            "name": "planet",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0, "roughnessFactor": 0.95},
            "normalTexture": {"index": 1}}],
        "textures": [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}],
        "images": [{"bufferView": v_img0, "mimeType": "image/png"},
                   {"bufferView": v_img1, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987,
                      "wrapS": 10497, "wrapT": 10497}],
        "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    while len(blob) % 4:
        blob.append(0)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<I", len(js)) + b"glTF" + js)
        fh.write(struct.pack("<I", len(blob)) + b"glTF" + bytes(blob))


# ---------------------------------------------------------------------------
# Compute shaders (GPU terrain + thermal erosion)
# ---------------------------------------------------------------------------

_COMPUTE_TERRAIN = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly restrict buffer DBuf { float dirs[]; };
layout(std430, binding = 1) writeonly restrict buffer EBuf { float elev[]; };
layout(std430, binding = 2) writeonly restrict buffer GBuf { float glow[]; };
uniform float uN, uCScale, uSea, uMtn, uWarp, uDetail, uNCr, uNVo, uLava;
uniform vec3 uSeed;
uniform vec4 uCr[100];
uniform vec4 uVo[40];
uniform vec2 uVoH[40];

vec3 mod289(vec3 x){return x - floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x - floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
float fbm4(vec3 p){ float a=0.5, s=0.0;
  for(int i=0;i<4;i++){ s+=a*snoise(p); p=p.zxy*2.03+vec3(11.3,7.7,3.1); a*=0.5; }
  return s/0.9375; }
float fbm3(vec3 p){ float a=0.5, s=0.0;
  for(int i=0;i<3;i++){ s+=a*snoise(p); p=p.zxy*2.03+vec3(5.2,1.3,7.9); a*=0.5; }
  return s/0.875; }
float fbm6(vec3 p){ float a=0.5, s=0.0;
  for(int i=0;i<6;i++){ s+=a*snoise(p); p=p.zxy*2.03+vec3(3.7,9.1,1.3); a*=0.5; }
  return s/0.984375; }
float ridged6(vec3 p){ float a=0.5, prev=1.0, s=0.0;
  for(int i=0;i<6;i++){ float n=1.0-abs(snoise(p)); n=n*n*prev;
    prev=clamp(n*2.0,0.0,1.0); s+=n*a; p=p.zxy*2.1+vec3(2.2,5.5,8.8); a*=0.5; }
  return s/0.984375; }
float ss(float a, float b, float x){ float t=clamp((x-a)/(b-a),0.0,1.0);
  return t*t*(3.0-2.0*t); }

void main(){
  uint gid = gl_GlobalInvocationID.x;
  if (gid >= uint(uN)) return;
  int i = int(gid);
  vec3 d = vec3(dirs[3*i], dirs[3*i+1], dirs[3*i+2]);
  vec3 x = d;
  vec3 w = fbm3(x + uSeed) * uWarp;
  vec3 q = x + w * vec3(1.0, 0.9, 1.1);
  float land = fbm6(q * uCScale + uSeed * 0.7) - uSea;
  float region = fbm3(q * 1.3 + vec3(7.7, 3.9, 5.2) + uSeed);
  float mask = ss(0.02, 0.22, land) * ss(-0.05, 0.35, region);
  float mtn = mask * ridged6(q * 2.6 + vec3(11.7, 5.1, 9.4)) * uMtn;
  float ln = ss(0.0, 0.12, land);
  float hills = fbm4(q * 7.0 + vec3(21.3, 13.7, 31.9)) * (0.012 + 0.055*ln) * uDetail;
  float e = land * 0.22 + mtn + hills;
  float deep = ss(0.03, 0.16, -land);
  float spine = ridged6(q * 1.05 + vec3(31.7, 17.3, 23.9));
  e = (land < 0.0) ? min(e + spine * deep * 0.05, -0.003) : e;
  float g = 0.0;
  for (int c = 0; c < 100; c++) {
    if (float(c) >= uNCr) break;
    vec4 cr = uCr[c];
    float t = length(d - cr.xyz) / cr.w;
    float tt = min(t, 1.0);
    e += (1.0-tt*tt)*(1.0-tt*tt)*cr.w*0.55
       + exp(-pow((t-1.0)/0.22, 2.0))*cr.w*0.30;
  }
  for (int v = 0; v < 40; v++) {
    if (float(v) >= uNVo) break;
    vec4 vo = uVo[v];
    float t = length(d - vo.xyz) / vo.w;
    e += pow(max(1.0-t, 0.0), 1.4) * uVoH[v].x * (1.0 - 0.5*exp(-pow(t/0.16, 2.0)));
    g += exp(-pow(t/0.10, 2.0)) * 0.9;
  }
  elev[i] = e;
  glow[i] = uLava * g;
}"""

_COMPUTE_THERMAL = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly restrict buffer EA { float ea[]; };
layout(std430, binding = 1) writeonly restrict buffer EB { float eb[]; };
layout(std430, binding = 2) readonly restrict buffer NB { int nb[]; };
uniform float uN, uK;
void main(){
  uint gid = gl_GlobalInvocationID.x;
  if (gid >= uint(uN)) return;
  int i = int(gid);
  float h = ea[i];
  float mn = h, sum = 0.0;
  int K = int(uK);
  for (int k = 0; k < K; k++) {
    float hn = ea[nb[i*K + k]];
    mn = min(mn, hn);
    sum += hn;
  }
  float f = clamp((h - mn - 0.006) * 8.0, 0.0, 1.0) * 0.55;
  eb[i] = h + f * (sum / uK - h);
}"""


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

RING_GLSL = """
uniform float uSeedPh;
#define RING_IN  1.55
#define RING_OUT 2.85
float ringAlpha(float r) {
    float t = clamp((r - RING_IN) / (RING_OUT - RING_IN), 0.0, 1.0);
    float n = sin(t * 61.0 + uSeedPh) * 0.45
            + sin(t * 127.0 + uSeedPh * 1.7) * 0.30
            + sin(t * 251.0 + uSeedPh * 2.3) * 0.25;
    float a = 0.32 + 0.55 * n + 0.22 * sin(t * 19.0 + uSeedPh);
    a *= smoothstep(0.0, 0.05, t) * (1.0 - smoothstep(0.95, 1.0, t));
    a *= 0.30 + 0.70 * smoothstep(0.015, 0.08, abs(t - 0.60));
    return clamp(a, 0.0, 1.0);
}
"""

CYC_GLSL = """
uniform vec4 uCyc[6];
uniform float uCycN;
uniform float uCycT;
vec3 swirlDir(vec3 d) {
    for (int i = 0; i < 6; i++) {
        if (float(i) >= uCycN) break;
        vec4 c = uCyc[i];
        vec3 ax = normalize(c.xyz);
        float da = dot(d, ax);
        vec3 tg = d - ax * da;
        float r = length(tg);
        float infl = exp(-(r*r)/(c.w*c.w + 1e-6));
        float ang = uCycT * 2.4 * sign(c.w) * infl;
        d = normalize(ax * da + cos(ang)*tg + sin(ang)*cross(ax, tg));
    }
    return d;
}
"""

PLANET_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNrm;
layout (location = 2) in vec3 aCol;
layout (location = 3) in float aWat;
layout (location = 4) in float aEmis;
layout (location = 5) in float aH;
layout (location = 6) in float aAO;
layout (location = 7) in float aRough;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vNrm; out vec3 vCol; out float vWat; out float vEmis;
out vec3 vPos; out vec3 vLocal; out float vH; out float vAO; out float vRough;
void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vPos = wp.xyz; vLocal = aPos;
    vNrm = mat3(uModel) * aNrm;
    vCol = aCol; vWat = aWat; vEmis = aEmis;
    vH = aH; vAO = aAO; vRough = aRough;
    gl_Position = uMVP * vec4(aPos, 1.0);
}"""

PLANET_FS = """#version 330 core
in vec3 vNrm; in vec3 vCol; in float vWat; in float vEmis;
in vec3 vPos; in vec3 vLocal; in float vH; in float vAO; in float vRough;
uniform vec3  uLightDir;
uniform vec3  uLLight;
uniform vec3  uEye;
uniform vec3  uGlowColor;
uniform vec3  uSkyTint;
uniform float uNight;
uniform float uTime;
uniform float uRingShade;
uniform samplerCube uCloudTex;
uniform float uEdge;
uniform float uCloudRot;
uniform float uCloudShadow;
uniform sampler2D uRelTex;
uniform float uRelStrength;
uniform vec4  uMoonPos[3];
uniform float uMoonN;
out vec4 fragColor;
""" + RING_GLSL + CYC_GLSL + """
vec3 rotY(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
}
void main() {
    vec3 Ng = normalize(vNrm);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uEye - vPos);
    vec3 N = Ng;
    if (vWat < 0.5 && uRelStrength > 0.001) {
        float rs = uRelStrength * mix(0.3, 1.0, vRough);
        vec3 w = pow(abs(Ng), vec3(4.0));
        w /= max(w.x + w.y + w.z, 1e-4);
        vec3 g = vec3(0.0);
        vec3 t = texture(uRelTex, vLocal.xy * 18.0).rgb * 2.0 - 1.0;
        g += w.z * vec3(t.x, t.y, 0.0);
        t = texture(uRelTex, vLocal.zy * 18.0).rgb * 2.0 - 1.0;
        g += w.x * vec3(0.0, t.x, t.y);
        t = texture(uRelTex, vLocal.xz * 18.0).rgb * 2.0 - 1.0;
        g += w.y * vec3(t.x, 0.0, t.y);
        N = normalize(Ng - rs * g);
    }
    float diff = max(dot(N, L), 0.0);
    if (uCloudShadow > 0.5) {
        vec3 cd = rotY(normalize(vLocal), -uCloudRot);
        cd = swirlDir(cd);
        float d = textureLod(uCloudTex, cd, 0.0).r;
        diff *= 1.0 - 0.45 * smoothstep(uEdge - 0.06, uEdge + 0.14, d);
    }
    if (uRingShade > 0.5 && abs(uLLight.y) > 1e-4) {
        float t = -vLocal.y / uLLight.y;
        if (t > 0.0) {
            float r = length(vLocal + t * uLLight);
            diff *= 1.0 - 0.72 * ringAlpha(r);
        }
    }
    for (int i = 0; i < 3; i++) {
        if (float(i) >= uMoonN) break;
        vec3 d = uMoonPos[i].xyz - vLocal;
        float tp = dot(d, uLLight);
        if (tp > 0.0) {
            float dd = length(d - tp * uLLight);
            float occ = 1.0 - smoothstep(uMoonPos[i].w * 0.7, uMoonPos[i].w * 1.5, dd);
            diff *= 1.0 - 0.85 * occ;
        }
    }
    // land + seafloor (ocean surface is a separate water-plane pass)
    vec3 col = vCol * (mix(0.05, 0.42, vWat) * vAO + 1.10 * diff)
             + vec3(0.008, 0.012, 0.030);
    float foam = vWat * smoothstep(0.014, 0.001, abs(vH))
               * (0.55 + 0.45 * sin(uTime * 2.3 + vLocal.x * 90.0 + vLocal.z * 70.0));
    col += vec3(0.95, 0.97, 1.0) * foam * 0.22 * (0.2 + 0.8 * diff);
    float fl = 0.85 + 0.15 * sin(uTime * 6.0 + vLocal.x * 537.0
                                 + vLocal.y * 311.0 + vLocal.z * 419.0);
    col += vec3(1.0, 0.72, 0.35) * vEmis * uNight * fl * 1.8
         * smoothstep(0.12, -0.18, dot(Ng, L));
    col += uGlowColor * vEmis;
    fragColor = vec4(col, 1.0);
}"""

WATER_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vPos; out vec3 vLocal;
void main() {
    vLocal = aPos;
    vPos = (uModel * vec4(aPos, 1.0)).xyz;
    gl_Position = uMVP * vec4(aPos, 1.0);
}"""

WATER_FS = """#version 330 core
in vec3 vPos; in vec3 vLocal;
uniform sampler2D uScene;
uniform vec2 uRes;
uniform vec3 uEye;
uniform vec3 uLight;
uniform vec3 uSkyTint;
uniform vec3 uDeep;
uniform vec3 uGlow;
uniform float uTime;
uniform samplerCube uCloudTex;
uniform float uEdge;
uniform float uCloudRot;
uniform float uCloudShadow;
uniform mat4 uModel;
out vec4 fragColor;
""" + CYC_GLSL + """
vec3 rotY(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
}
void main() {
    vec3 Ng = normalize(mat3(uModel) * normalize(vLocal));
    vec3 L = normalize(uLight);
    vec3 V = normalize(uEye - vPos);
    float t1 = uTime * 1.6, t2 = uTime * 2.4;
    vec3 w1 = vec3(sin(vLocal.x*41.0+t1), sin(vLocal.y*37.0+t1*0.8), sin(vLocal.z*43.0+t2));
    vec3 w2 = vec3(sin(vLocal.x*97.0-t2*1.3), sin(vLocal.y*89.0+t2), sin(vLocal.z*101.0+t1*1.1));
    vec3 N = normalize(Ng + 0.030*w1 + 0.014*w2);
    float ndv = max(dot(N, V), 0.0);
    float F = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    float diff = max(dot(N, L), 0.0);
    if (uCloudShadow > 0.5) {
        vec3 cd = rotY(normalize(vLocal), -uCloudRot);
        cd = swirlDir(cd);
        float d = textureLod(uCloudTex, cd, 0.0).r;
        diff *= 1.0 - 0.45 * smoothstep(uEdge - 0.06, uEdge + 0.14, d);
    }
    vec2 uv = gl_FragCoord.xy / uRes + (w1.xy + w2.yz) * 0.010;
    vec3 refr = texture(uScene, clamp(uv, 0.001, 0.999)).rgb;
    refr = mix(refr, refr * uDeep * 2.2, 0.62) * (0.35 + 0.65 * diff);
    vec3 refl = uSkyTint * (0.30 + 0.90 * diff);
    vec3 col = mix(refr, refl, 0.88 * F);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 140.0) * diff;
    col += vec3(1.2, 1.3, 1.45) * spec * (0.35 + 0.65 * F) * 1.4;
    col += uGlow;
    fragColor = vec4(col, 1.0);
}"""

TREE_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNrm;
layout (location = 2) in vec3 iPos;
layout (location = 3) in float iRot;
layout (location = 4) in float iSca;
layout (location = 5) in vec3 iCol;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform float uFade;
out vec3 vN; out vec3 vCol;
void main() {
    float c = cos(iRot), s = sin(iRot);
    vec3 p = vec3(c*aPos.x + s*aPos.z, aPos.y, -s*aPos.x + c*aPos.z);
    vec3 n = vec3(c*aNrm.x + s*aNrm.z, aNrm.y, -s*aNrm.x + c*aNrm.z);
    vN = mat3(uModel) * n;
    vCol = iCol;
    gl_Position = uMVP * vec4(iPos + p * iSca * uFade, 1.0);
}"""

TREE_FS = """#version 330 core
in vec3 vN; in vec3 vCol;
uniform vec3 uLightDir;
out vec4 fragColor;
void main() {
    float d = max(dot(normalize(vN), normalize(uLightDir)), 0.0);
    fragColor = vec4(vCol * (0.08 + 1.05 * d) + vec3(0.005, 0.010, 0.008), 1.0);
}"""

RING_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uVP; uniform mat4 uModel;
out vec3 vW; out vec3 vL;
void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vW = wp.xyz; vL = aPos;
    gl_Position = uVP * wp;
}"""

RING_FS = """#version 330 core
in vec3 vW; in vec3 vL;
uniform vec3 uLightDir; uniform mat4 uModel; uniform float uOpacity;
out vec4 fragColor;
""" + RING_GLSL + """
void main() {
    float r = length(vL.xz);
    float a0 = ringAlpha(r);
    float a = a0 * uOpacity;
    if (a < 0.004) discard;
    vec3 L = normalize(uLightDir);
    vec3 N = normalize(mat3(uModel) * vec3(0.0, 1.0, 0.0));
    float li = 0.20 + 0.80 * abs(dot(N, L));
    float b = dot(vW, L);
    float c = dot(vW, vW) - 1.0;
    if (b * b - c > 0.0 && b < 0.0) li *= 0.10;
    vec3 base = mix(vec3(0.52, 0.47, 0.41), vec3(0.90, 0.86, 0.78), a0);
    fragColor = vec4(base * li * 1.35, a);
}"""

CLOUD_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uMVP; uniform mat4 uModel; uniform float uRadius;
out vec3 vDir; out vec3 vW;
void main() {
    vDir = aPos;
    vW = (uModel * vec4(aPos * uRadius, 1.0)).xyz;
    gl_Position = uMVP * vec4(aPos * uRadius, 1.0);
}"""

CLOUD_FS = """#version 330 core
in vec3 vDir; in vec3 vW;
uniform samplerCube uCloudTex;
uniform vec3 uLightDir; uniform vec3 uCloudColor; uniform vec3 uEye;
uniform float uEdge; uniform float uCloudMax; uniform float uDrift;
out vec4 fragColor;
""" + CYC_GLSL + """
vec3 rotY(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
}
void main() {
    vec3 d = rotY(normalize(vDir), -uDrift);
    d = swirlDir(d);
    float v = texture(uCloudTex, d).r;
    float a = smoothstep(uEdge - 0.10, uEdge + 0.10, v) * uCloudMax;
    if (a < 0.01) discard;
    vec3 N = normalize(vDir);
    float diff = max(dot(N, normalize(uLightDir)), 0.0);
    fragColor = vec4(uCloudColor * (0.05 + 1.00 * diff) + vec3(0.015, 0.02, 0.04), a);
}"""

# volumetric clouds: march the cubemap density through the shell slab
CLOUDVOL_FS = """#version 330 core
in vec3 vDir; in vec3 vW;
uniform samplerCube uCloudTex;
uniform vec3 uEye; uniform vec3 uLight; uniform vec3 uCloudColor;
uniform float uEdge; uniform float uDrift; uniform float uR0; uniform float uR1;
out vec4 fragColor;
""" + CYC_GLSL + """
vec3 rotY(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
}
float dens(vec3 d) {
    d = rotY(d, -uDrift);
    d = swirlDir(d);
    return smoothstep(uEdge - 0.10, uEdge + 0.10, texture(uCloudTex, d).r);
}
void main() {
    vec3 ro = uEye;
    vec3 rd = normalize(vW - uEye);
    float b = dot(ro, rd);
    float h0 = b*b - (dot(ro,ro) - uR0*uR0);
    float t0 = 0.0;
    if (h0 > 0.0) t0 = max(-b - sqrt(h0), 0.0);
    float h1 = b*b - (dot(ro,ro) - uR1*uR1);
    if (h1 < 0.0) { fragColor = vec4(0.0); return; }
    float t1 = -b + sqrt(h1);
    if (t1 <= t0) { fragColor = vec4(0.0); return; }
    vec3 L = normalize(uLight);
    float mu = dot(rd, L);
    float ph = 0.35 + 0.65 * max(0.0, 0.5 + 0.5 * mu);
    float stepL = (t1 - t0) / 14.0;
    float trans = 1.0; vec3 acc = vec3(0.0); float aAcc = 0.0;
    for (int i = 0; i < 14; i++) {
        vec3 p = ro + rd * (t0 + stepL * (float(i) + 0.5));
        float d = dens(normalize(p));
        if (d > 0.01) {
            float od = 0.0;
            for (int j = 0; j < 4; j++)
                od += dens(normalize(p + L * 0.05 * (float(j) + 0.5))) * 0.05;
            float lit = exp(-od * 6.0);
            float a = 1.0 - exp(-d * stepL * 7.0);
            vec3 cloc = uCloudColor * (0.10 + 1.05 * lit * ph) + vec3(0.02, 0.025, 0.05);
            acc += cloc * a * trans;
            aAcc += a * trans;
            trans *= 1.0 - a;
            if (trans < 0.05) break;
        }
    }
    if (aAcc < 0.004) { fragColor = vec4(0.0); return; }
    fragColor = vec4(acc / aAcc, aAcc);
}"""

ATMOQ_VS = """#version 330 core
layout (location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}"""

ATMOQ_FS = """#version 330 core
in vec2 vUV;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform vec3 uLight;
uniform vec3 uBetaR;
uniform float uDensity;
uniform float uSunI;
uniform float uAtmoR;
uniform float uTime;
uniform float uAurora;
uniform vec3 uAurAxis;
uniform vec3 uAurX;
uniform vec3 uAurY;
uniform vec3 uAurColor;
uniform float uMulti;
out vec4 fragColor;
const float HR = 0.11;
const float HM = 0.045;
bool raySphere(vec3 ro, vec3 rd, float r, out float t0, out float t1) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    t0 = -b - h; t1 = -b + h;
    return t1 > 0.0;
}
vec2 lightOD(vec3 p, vec3 L) {
    float t0, t1;
    if (!raySphere(p, L, uAtmoR, t0, t1)) return vec2(-1.0);
    float q0, q1;
    if (raySphere(p, L, 1.0, q0, q1) && q0 > 0.0) return vec2(-1.0);
    float odR = 0.0, odM = 0.0;
    float ds = t1 / float(SEC);
    for (int i = 0; i < SEC; i++) {
        vec3 s = p + L * ds * (float(i) + 0.5);
        float h = max(length(s) - 1.0, 0.0);
        odR += exp(-h / HR) * ds;
        odM += exp(-h / HM) * ds;
    }
    return vec2(odR, odM);
}
void main() {
    vec4 pf = uInvVP * vec4(vUV * 2.0 - 1.0, 1.0, 1.0);
    vec4 pn = uInvVP * vec4(vUV * 2.0 - 1.0, -1.0, 1.0);
    vec3 ro = uEye;
    vec3 rd = normalize(pf.xyz / pf.w - pn.xyz / pn.w);
    float t0, t1;
    if (!raySphere(ro, rd, uAtmoR, t0, t1)) { fragColor = vec4(0.0); return; }
    t0 = max(t0, 0.0);
    float p0, p1;
    bool hitP = raySphere(ro, rd, 1.0, p0, p1);
    float tEnd = (hitP && p0 > 0.0) ? min(p0, t1) : t1;
    if (tEnd <= t0) { fragColor = vec4(0.0); return; }
    vec3 L = normalize(uLight);
    float ds = (tEnd - t0) / float(PRI);
    vec3 betaM = vec3(0.045) * uDensity;
    float odRv = 0.0, odMv = 0.0;
    vec3 sumR = vec3(0.0), sumM = vec3(0.0), msA = vec3(0.0);
    float aur = 0.0;
    float mu = dot(rd, L);
    float phR = 0.0596 * (1.0 + mu * mu);
    float g = 0.76;
    float phM = (1.0 - g*g) / (12.56637 * pow(1.0 + g*g - 2.0*g*mu, 1.5));
    for (int i = 0; i < PRI; i++) {
        vec3 p = ro + rd * (t0 + ds * (float(i) + 0.5));
        float h = max(length(p) - 1.0, 0.0);
        float dR = exp(-h / HR);
        float dM = exp(-h / HM);
        odRv += dR * ds;
        odMv += dM * ds;
        vec2 lod = lightOD(p, L);
        if (lod.x >= 0.0) {
            vec3 att = exp(-uBetaR * (odRv + lod.x) - betaM * (odMv + lod.y));
            sumR += dR * ds * att;
            sumM += dM * ds * att;
            msA += att * dR * ds;
        }
        if (uAurora > 0.5) {
            vec3 pn_ = normalize(p);
            float sy = abs(dot(pn_, uAurAxis));
            float pol = smoothstep(0.55, 0.75, sy) * (1.0 - smoothstep(0.92, 1.0, sy));
            if (pol > 0.001) {
                float lon = atan(dot(pn_, uAurY), dot(pn_, uAurX));
                float w1 = sin(lon*5.0 + uTime*0.9 + 2.5*sin(lon*2.0 - uTime*0.37));
                float w2 = sin(lon*9.0 - uTime*1.3 + 3.0*sin(lon*3.0 + uTime*0.21) + 1.7);
                float curt = pow(0.5+0.5*w1, 3.0)*0.7 + pow(0.5+0.5*w2, 4.0)*0.5;
                float alt = exp(-pow((h - 0.06)/0.05, 2.0));
                float night = 1.0 - smoothstep(0.06, -0.14, dot(pn_, L));
                aur += pol * curt * alt * night * ds * 2.2;
            }
        }
    }
    vec3 col = uSunI * (sumR * uBetaR * phR + sumM * betaM * phM);
    col += uSunI * uMulti * 0.45 * msA * uBetaR;
    col += uAurColor * aur * 1.5;
    fragColor = vec4(col, 1.0);
}"""

ADD_FS = """#version 330 core
uniform sampler2D uTex;
in vec2 vUV;
out vec4 o;
void main() { o = vec4(texture(uTex, vUV).rgb, 1.0); }"""

GODRAY_FS = """#version 330 core
uniform sampler2D uTex;
uniform vec2 uSunUV;
uniform float uOccl;
in vec2 vUV;
out vec4 o;
void main() {
    vec2 uv = vUV;
    vec2 dir = (uSunUV - vUV) * (0.9 / 48.0);
    float w = 1.0;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 48; i++) {
        uv += dir;
        w *= 0.955;
        acc += texture(uTex, clamp(uv, 0.0, 1.0)).rgb * w;
    }
    o = vec4(acc / 48.0 * vec3(1.0, 0.82, 0.55) * 2.6 * uOccl, 1.0);
}"""

SUNBODY_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNrm;
uniform mat4 uMVP; uniform mat4 uModel;
out vec3 vN; out vec3 vL;
void main() {
    vL = aPos;
    vN = (uModel * vec4(aNrm, 0.0)).xyz;
    gl_Position = uMVP * vec4(aPos, 1.0);
}"""

SUNBODY_FS = """#version 330 core
in vec3 vN; in vec3 vL;
uniform mat4 uModel;
uniform vec3 uEye;
uniform float uTime;
out vec4 fragColor;
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i), hash13(i+vec3(1,0,0)), f.x),
                   mix(hash13(i+vec3(0,1,0)), hash13(i+vec3(1,1,0)), f.x), f.y),
               mix(mix(hash13(i+vec3(0,0,1)), hash13(i+vec3(1,0,1)), f.x),
                   mix(hash13(i+vec3(0,1,1)), hash13(i+vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.1; a *= 0.5; }
    return s;
}
void main() {
    float c = cos(uTime * 0.05), s = sin(uTime * 0.05);
    vec3 p = vec3(c*vL.x + s*vL.z, vL.y, -s*vL.x + c*vL.z);
    float n = fbm(p * 5.0);
    vec3 col = mix(vec3(1.0, 0.40, 0.03), vec3(1.0, 0.95, 0.78),
                   smoothstep(-0.15, 0.55, n));
    vec3 wp = (uModel * vec4(vL, 1.0)).xyz;
    float mu = max(dot(normalize(vN), normalize(uEye - wp)), 0.0);
    col *= (0.45 + 0.55 * mu);          // limb darkening
    fragColor = vec4(col * 6.0, 1.0);
}"""

SUN_VS = """#version 330 core
layout (location = 0) in vec2 aCorner;
uniform mat4 uVP; uniform vec3 uCenter; uniform vec3 uEye; uniform float uSize;
out vec2 vUV;
void main() {
    vec3 toEye = normalize(uEye - uCenter);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), toEye));
    vec3 up = cross(toEye, right);
    vec3 wp = uCenter + (right * aCorner.x + up * aCorner.y) * uSize;
    vUV = aCorner;
    gl_Position = uVP * vec4(wp, 1.0);
}"""

SUN_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
void main() {
    float r = length(vUV);
    float halo = exp(-r * 2.6) * 0.7;
    fragColor = vec4(vec3(1.0, 0.75, 0.45) * halo, 1.0);
}"""

ATMOSHELL_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uVP; uniform mat4 uModel;
out vec3 vN; out vec3 vW;
void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vW = wp.xyz;
    vN = mat3(uModel) * aPos;
    gl_Position = uVP * wp;
}"""

ATMOSHELL_FS = """#version 330 core
in vec3 vN; in vec3 vW;
uniform vec3 uEye; uniform vec3 uLight; uniform vec3 uCol;
out vec4 fragColor;
void main() {
    vec3 N = normalize(vN);
    vec3 V = normalize(uEye - vW);
    vec3 L = normalize(uLight);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    float sun = 0.25 + 0.75 * max(dot(N, L), 0.0);
    fragColor = vec4(uCol * rim * sun * 1.4, 1.0);
}"""

PRECIP_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uVP;
uniform vec3 uCenter;
uniform float uSize;
void main() {
    gl_Position = uVP * vec4(aPos + uCenter, 1.0);
    gl_PointSize = uSize;
}"""

PRECIP_FS = """#version 330 core
uniform vec3 uColor;
out vec4 fragColor;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    fragColor = vec4(uColor, 1.0);
}"""

STAR_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aSize;
layout (location = 2) in float aBright;
layout (location = 3) in vec3 aCol;
uniform mat4 uVP; uniform float uTime;
out float vBright; out vec3 vCol;
void main() {
    float h = fract(sin(dot(aPos.xy, vec2(12.9898, 78.233))) * 43758.5453);
    vBright = aBright * (0.70 + 0.30 * sin(uTime * 1.5 + h * 6.2831));
    vCol = aCol;
    gl_Position = uVP * vec4(aPos, 1.0);
    gl_PointSize = aSize;
}"""

STAR_FS = """#version 330 core
in float vBright; in vec3 vCol;
out vec4 fragColor;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    fragColor = vec4(vCol * vBright * 1.5, 1.0);
}"""

LINE_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uVP;
void main() { gl_Position = uVP * vec4(aPos, 1.0); }"""

LINE_FS = """#version 330 core
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }"""

MARK_VS = """#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uVP; uniform mat4 uModel;
void main() {
    gl_Position = uVP * uModel * vec4(aPos, 1.0);
    gl_PointSize = 11.0;
}"""

MARK_FS = """#version 330 core
out vec4 fragColor;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    fragColor = vec4(1.8, 0.8, 0.15, 1.0);
}"""

QUAD_VS = """#version 330 core
layout (location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}"""

BRIGHT_FS = """#version 330 core
uniform sampler2D uTex; uniform float uThresh;
in vec2 vUV; out vec4 o;
void main() {
    vec3 c = texture(uTex, vUV).rgb;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    o = vec4(c * max(l - uThresh, 0.0) / (l + 1e-4), 1.0);
}"""

BLUR_FS = """#version 330 core
uniform sampler2D uTex; uniform vec2 uStep;
in vec2 vUV; out vec4 o;
void main() {
    vec3 s = texture(uTex, vUV).rgb * 0.227027;
    s += (texture(uTex, vUV + uStep * 1.38461).rgb
        + texture(uTex, vUV - uStep * 1.38461).rgb) * 0.316216;
    s += (texture(uTex, vUV + uStep * 3.23077).rgb
        + texture(uTex, vUV - uStep * 3.23077).rgb) * 0.070270;
    o = vec4(s, 1.0);
}"""

COMPOSITE_FS = """#version 330 core
uniform sampler2D uScene; uniform sampler2D uBloom; uniform sampler2D uGod;
uniform float uStrength; uniform float uGStr; uniform float uExposure;
uniform vec2 uSunUV; uniform float uOccl; uniform float uAspect;
uniform float uFlare;
in vec2 vUV; out vec4 o;
vec3 aces(vec3 x) {
    return clamp((x * (2.51*x + 0.03)) / (x * (2.43*x + 0.59) + 0.14), 0.0, 1.0);
}
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
void main() {
    vec3 c = texture(uScene, vUV).rgb
           + texture(uBloom, vUV).rgb * uStrength
           + texture(uGod, vUV).rgb * uGStr;
    if (uSunUV.x > -5.0) {
        vec2 sc = (vUV - 0.5) * vec2(uAspect, 1.0);
        vec2 su = (uSunUV - 0.5) * vec2(uAspect, 1.0);
        vec3 fg = vec3(0.0);
        vec2 gp = mix(su, vec2(0.0), 0.35);
        fg += vec3(1.0, 0.6, 0.3) * smoothstep(0.06, 0.0, length(sc - gp)) * 0.35;
        gp = mix(su, vec2(0.0), 0.75);
        fg += vec3(0.4, 0.7, 1.0) * smoothstep(0.11, 0.0, length(sc - gp)) * 0.30;
        gp = mix(su, vec2(0.0), 1.35);
        fg += vec3(1.0, 0.8, 0.4) * smoothstep(0.05, 0.0, length(sc - gp)) * 0.25;
        c += fg * uFlare * uOccl;
    }
    c = aces(c * uExposure);
    c += (hash12(gl_FragCoord.xy) - 0.5) / 128.0;
    o = vec4(pow(clamp(c, 0.0, 1.0), vec3(0.4545)), 1.0);
}"""


# ---------------------------------------------------------------------------
# Tree geometry
# ---------------------------------------------------------------------------

def _tree_geometry():
    segs = 6
    verts, norms, idx = [], [], []
    for r, y0, y1 in ((0.55, 0.0, 0.62), (0.38, 0.45, 1.0)):
        for i in range(segs):
            a0 = 2.0 * math.pi * i / segs
            a1 = 2.0 * math.pi * (i + 1) / segs
            p0 = np.array([math.cos(a0) * r, y0, math.sin(a0) * r])
            p1 = np.array([math.cos(a1) * r, y0, math.sin(a1) * r])
            apex = np.array([0.0, y1, 0.0])
            n = np.cross(p1 - p0, apex - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            mid = (p0 + p1) / 2.0
            flip = (n @ np.array([mid[0], 0.0, mid[2]])) < 0
            base = len(verts)
            tri = [p1, p0, apex] if flip else [p0, p1, apex]
            verts += tri
            norms += [-n, -n, -n] if flip else [n, n, n]
            idx += [base, base + 1, base + 2]
    return (np.array(verts, np.float32), np.array(norms, np.float32),
            np.array(idx, np.uint32))


# ---------------------------------------------------------------------------
# OpenGL widget
# ---------------------------------------------------------------------------

RING_INNER, RING_OUTER = 1.55, 2.85
SS_FACTOR = 1.25
EXPOSURE = 1.15

MOONS = (
    (5.4,  0.010, 0.20, 0.0),
    (7.8,  0.016, 0.42, 2.2),
    (4.0, -0.013, 0.06, 4.4),
)
_MOON_SPECS = (
    (424242, 0.27, 85, (1.00, 1.00, 1.00)),
    (31337,  0.15, 60, (0.85, 0.95, 1.10)),
    (777,    0.11, 110, (1.12, 0.94, 0.84)),
)
_CUBE_FACES = (
    (( 1, 0, 0), (0, 0, -1), (0, -1, 0)),
    ((-1, 0, 0), (0, 0,  1), (0, -1, 0)),
    (( 0, 1, 0), (1, 0,  0), (0,  0, 1)),
    (( 0,-1, 0), (1, 0,  0), (0,  0,-1)),
    (( 0, 0, 1), (1, 0,  0), (0, -1, 0)),
    (( 0, 0,-1), (-1, 0, 0), (0, -1, 0)),
)
_SKYBOX_FACES = (
    ("px", (1, 0, 0), (0, -1, 0)), ("nx", (-1, 0, 0), (0, -1, 0)),
    ("py", (0, 1, 0), (0, 0, 1)),  ("ny", (0, -1, 0), (0, 0, -1)),
    ("pz", (0, 0, 1), (0, -1, 0)), ("nz", (0, 0, -1), (0, -1, 0)),
)


class Mesh:
    __slots__ = ("vao", "buffers", "count", "mode", "inst")

    def __init__(self, vao, buffers, count, mode, inst=0):
        self.vao = vao; self.buffers = buffers
        self.count = count; self.mode = mode; self.inst = inst


class PlanetWidget(QOpenGLWidget):
    probePicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.planet = None
        self.auto_rotate = True
        self.wireframe = False
        self.show_clouds = True
        self.show_atmo = True
        self.show_lights = True
        self.show_moon = True
        self.show_aurora = True
        self.show_orbits = True
        self.show_veg = True
        self.show_companion = False
        self.show_moon_rings = False
        self.show_moon_atmo = False
        self.show_weather = True
        self.use_gpu = True
        self.sun_cycle = False
        self.sun_cycle_speed = 4.0
        self.auto_hires = False

        self.tilt = 23.0
        self.cloud_cover = 0.55
        self.sun_az = 40.0
        self.sun_el = 15.0
        self.ring_opacity = 0.0
        self.moon_count = 1
        self.speed = 1.0
        self.bloom = 0.35
        self.god_strength = 0.35
        self.flare = 0.6
        self.multi = 0.30
        self.relief_strength = 0.28
        self.atmo_density = 1.3
        self.terraform = 0.0
        self.atmo_quality = "Medium"
        self.veg_density = 0.70
        self.subdiv_change_cb = None
        self.rotate_cb = None
        self.sun_ui_cb = None
        self.hover_cb = None
        self.fps_cb = None

        self.yaw, self.pitch, self.dist = 25.0, 12.0, 3.4
        self.spin = 0.0
        self.cloud_drift = 0.0
        self.moon_angle = 35.0
        self.time_s = 0.0
        self._last_t = None
        self._hires = False
        self._anim = None
        self._sky_mode = False
        self._hover_t = 0.0
        self._sun_uv = (-10.0, -10.0)
        self._sun_occl = 0.0

        self._planet_mesh = None
        self._tree_mesh = None
        self._tree_count = 0
        self._water_mesh = None
        self._cloud_mesh = None
        self._cloud_tex = None
        self._cloud_seed = None
        self._cyclones = []
        self._detail_tex = None
        self._moon_meshes = []
        self._shell_mesh = None
        self._sun_mesh = None
        self._corona_mesh = None
        self._precip_mesh = None
        self._precip_pos = None
        self._star_mesh = None
        self._ring_mesh = None
        self._marker_mesh = None
        self._orbit_meshes = []
        self._marker_on = False
        self._marker_pos = np.zeros(3, np.float32)
        self._dirs = None
        self._inv_vp = None
        self._eye = QVector3D(0, 0, 3)
        self._model_inv = QMatrix4x4()
        self.has_compute = False
        self._gpu_progs = {}

        self._fbo_size = None
        self._ss = (1, 1)
        self._half = (1, 1)
        self._fbo_ok = False
        self._hdr = True
        self._fbo_scene = self._fbo_a = self._fbo_b = self._fbo_atmo = None
        self._fbo_copy = None
        self._scene_tex = self._tex_a = self._tex_b = self._atmo_tex = None
        self._copy_tex = None
        self._depth_rb = None
        self._atmo_progs = {}

        self._fps_frames = 0
        self._fps_t = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    # ---- public ----
    def set_planet(self, data):
        self.planet = data
        self._marker_on = False
        self._dirs = None
        if self.isValid():
            self.makeCurrent()
            self._build_planet_meshes(data)
            self.doneCurrent()
        self.update()

    def set_atmo_quality(self, key):
        if key in _QUALITY:
            self.atmo_quality = key
            self.update()

    def _tick(self):
        now = time.perf_counter()
        dt = 0.016 if self._last_t is None else min(0.05, now - self._last_t)
        self._last_t = now
        self._fps_frames += 1
        self._fps_t += dt
        if self._fps_t >= 1.0:
            if self.fps_cb:
                self.fps_cb(int(round(self._fps_frames / self._fps_t)))
            self._fps_frames = 0
            self._fps_t = 0.0
        sdt = dt * self.speed
        self.time_s += sdt
        if self.auto_rotate:
            self.spin = (self.spin + 7.5 * sdt) % 360.0
        self.cloud_drift = (self.cloud_drift + 1.2 * sdt) % 360.0
        self.moon_angle = (self.moon_angle + 3.6 * sdt) % 360.0
        if self.sun_cycle:
            self.sun_az = (self.sun_az + self.sun_cycle_speed * sdt * 8.0) % 360.0
            if self.sun_ui_cb:
                self.sun_ui_cb(self.sun_az)
        if self._anim is not None:
            y0, p0, dy, dp, t = self._anim
            t = min(1.0, t + dt / 0.9)
            e = t * t * (3.0 - 2.0 * t)
            self.yaw = (y0 + dy * e) % 360.0
            self.pitch = p0 + dp * e
            if t >= 1.0:
                self._anim = None
        if self.auto_hires and self.subdiv_change_cb is not None:
            if not self._hires and self.dist < 2.25:
                self._hires = True
                self.subdiv_change_cb(True)
            elif self._hires and self.dist > 2.55:
                self._hires = False
                self.subdiv_change_cb(False)
        # precipitation
        if self._precip_pos is not None and self.show_weather:
            st = self.planet["style"] if self.planet else "terran"
            spd = 0.45 if st == "ice" else 2.2
            self._precip_pos[:, 1] -= spd * dt
            self._precip_pos[:, 0] += 0.15 * dt * (1 if st != "ice" else 0.3)
            self._precip_pos[:, 1] = np.where(self._precip_pos[:, 1] < -1.5,
                                              1.5, self._precip_pos[:, 1])
        self.update()

    def focus_latlon(self, lat, lon):
        if self.planet is None:
            return
        la, lo = math.radians(lat), math.radians(lon)
        p = np.array([math.cos(la) * math.cos(lo), math.sin(la),
                      math.cos(la) * math.sin(lo)], np.float64)
        if self._dirs is not None:
            d = self._dirs - p.astype(np.float32)
            i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
            self._marker_pos = self.planet["positions"][i].copy()
            self._marker_on = True
        model = QMatrix4x4()
        model.rotate(self.tilt, 0.0, 0.0, 1.0)
        model.rotate(self.spin, 0.0, 1.0, 0.0)
        w = model.map(QVector3D(*p))
        if w.length() < 1e-9:
            return
        w.normalize()
        yaw_t = math.degrees(math.atan2(-w.x(), w.z()))
        a = math.radians(yaw_t)
        vy = w.y()
        vz = -math.sin(a) * w.x() + math.cos(a) * w.z()
        pitch_t = max(-89.0, min(89.0, math.degrees(math.atan2(-vy, -vz))))
        dyaw = ((yaw_t - self.yaw + 180.0) % 360.0) - 180.0
        self._anim = [self.yaw, self.pitch, dyaw, pitch_t - self.pitch, 0.0]
        self.auto_rotate = False
        if self.rotate_cb:
            self.rotate_cb(False)

    def export_skybox(self, folder, size=1024):
        self.makeCurrent()
        tex = self._tex2d(size, size, False)
        fb, rb, ok = self._fbo_with(tex, size, size, depth=True)
        if not ok:
            GL.glDeleteTextures([tex])
            GL.glDeleteFramebuffers(1, [fb])
            GL.glDeleteRenderbuffers(1, [rb])
            self.doneCurrent()
            return []
        saved = (self._marker_on, self.show_orbits, self.show_clouds,
                 self.show_atmo, self.auto_rotate, self._sky_mode,
                 self.show_companion, self.show_veg)
        self._marker_on = False
        self.show_orbits = False
        self.show_clouds = False
        self.show_atmo = False
        self.show_companion = False
        self.show_veg = False
        self.auto_rotate = False
        self._sky_mode = True
        proj = QMatrix4x4()
        proj.perspective(90.0, 1.0, 0.1, 200.0)
        paths = []
        try:
            for name, fwd, up in _SKYBOX_FACES:
                view = QMatrix4x4()
                view.lookAt(QVector3D(0, 0, 0),
                            QVector3D(*fwd).normalized(), QVector3D(*up))
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb)
                GL.glViewport(0, 0, size, size)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                self._draw_scene(proj * view, QVector3D(0, 0, 0),
                                 self._light_vec())
                GL.glFinish()
                buf = GL.glReadPixels(0, 0, size, size, GL.GL_RGBA,
                                      GL.GL_UNSIGNED_BYTE)
                arr = np.frombuffer(buf, np.uint8).reshape(size, size, 4)
                arr = arr[::-1, :, :3].copy()
                img = QImage(arr.tobytes(), size, size, 3 * size,
                             QImage.Format_RGB888)
                p = os.path.join(folder, f"{name}.png")
                img.save(p, "PNG")
                paths.append(p)
        finally:
            (self._marker_on, self.show_orbits, self.show_clouds,
             self.show_atmo, self.auto_rotate, self._sky_mode,
             self.show_companion, self.show_veg) = saved
            GL.glDeleteFramebuffers(1, [fb])
            GL.glDeleteTextures([tex])
            GL.glDeleteRenderbuffers(1, [rb])
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            self.doneCurrent()
            self.update()
        return paths

    # ---- GPU compute path ----
    def _gpu_prog(self, key, src):
        prog = self._gpu_progs.get(key)
        if prog is None:
            prog = QOpenGLShaderProgram(self)
            if not (prog.addShaderFromSourceCode(QOpenGLShader.Compute, src)
                    and prog.link()):
                raise RuntimeError("Compute shader error:\n" + prog.log())
            self._gpu_progs[key] = prog
        return prog

    def gpu_terrain(self, p, verts):
        """Elevation (+lava glow) via compute shader; thermal erosion on GPU.
        Returns (elev, glow) or None if unsupported."""
        if not self.has_compute or not self.use_gpu or p.style == "gas":
            return None
        n = len(verts)
        groups = (n + 63) // 64
        self.makeCurrent()
        bufs = []
        try:
            dirs = verts.astype(np.float32).ravel()
            bdirs = GL.glGenBuffers(1); bufs.append(bdirs)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, bdirs)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, dirs.nbytes, dirs,
                            GL.GL_DYNAMIC_DRAW)
            bea = GL.glGenBuffers(1); bufs.append(bea)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, bea)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n * 4, None,
                            GL.GL_DYNAMIC_DRAW)
            beb = GL.glGenBuffers(1); bufs.append(beb)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, beb)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n * 4, None,
                            GL.GL_DYNAMIC_DRAW)
            bem = GL.glGenBuffers(1); bufs.append(bem)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, bem)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n * 4, None,
                            GL.GL_DYNAMIC_DRAW)

            tp = self._gpu_prog("terrain", _COMPUTE_TERRAIN)
            tp.bind()
            tp.setUniformValue("uN", float(n))
            tp.setUniformValue("uCScale", float(p.continent_scale))
            tp.setUniformValue("uSea", float(p.sea_level))
            tp.setUniformValue("uMtn", float(p.mountain_height))
            tp.setUniformValue("uWarp", float(p.warp))
            tp.setUniformValue("uDetail", float(p.detail))
            tp.setUniformValue("uNCr", float(int(p.impacts)))
            tp.setUniformValue("uNVo", float(int(p.volcanoes)))
            tp.setUniformValue("uLava", 1.0 if p.style == "lava" else 0.0)
            rng = np.random.default_rng(p.seed)
            tp.setUniformValue("uSeed", QVector3D(*(
                (rng.random(3) * 40.0 - 20.0).tolist())))
            cr_c, cr_r, cr_d = _crater_list(p)
            crv = [QVector4D(*(cr_c[i] * 1.0), cr_r[i]) for i in range(len(cr_r))]
            crv += [QVector4D(0, 0, 0, 1)] * (100 - len(crv))
            tp.setUniformValueArray("uCr", crv)
            vo_c, vo_r, vo_h = _volcano_list(p)
            vov = [QVector4D(*(vo_c[i] * 1.0), vo_r[i]) for i in range(len(vo_r))]
            vov += [QVector4D(0, 0, 0, 1)] * (40 - len(vov))
            vh = [QVector2D(vo_h[i], 0.0) for i in range(len(vo_h))]
            vh += [QVector2D(0, 0)] * (40 - len(vh))
            tp.setUniformValueArray("uVo", vov)
            tp.setUniformValueArray("uVoH", vh)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, bdirs)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, bea)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, bem)
            GL.glDispatchCompute(groups, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            tp.release()

            glow = None
            if p.style == "lava":
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, bem)
                glow = np.frombuffer(
                    GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, n * 4),
                    np.float32).copy()

            thermal = int(round(p.thermal + p.age * 0.08))
            if thermal > 0:
                _, faces = get_icosphere(p.subdiv)
                nb = build_neighbors(faces, n).astype(np.int32).ravel()
                bnb = GL.glGenBuffers(1); bufs.append(bnb)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, bnb)
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, nb.nbytes, nb,
                                GL.GL_DYNAMIC_DRAW)
                ep = self._gpu_prog("thermal", _COMPUTE_THERMAL)
                ep.bind()
                ep.setUniformValue("uN", float(n))
                ep.setUniformValue("uK", float(nb.size // n))
                cur, oth = bea, beb
                for _ in range(thermal):
                    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, cur)
                    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, oth)
                    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, bnb)
                    GL.glDispatchCompute(groups, 1, 1)
                    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                    cur, oth = oth, cur
                ep.release()

            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, cur)
            GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
            elev = np.frombuffer(
                GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, n * 4),
                np.float32).copy()
            for b in bufs:
                GL.glDeleteBuffers(1, [b])
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        except Exception:
            traceback.print_exc()
            for b in bufs:
                try:
                    GL.glDeleteBuffers(1, [b])
                except Exception:
                    pass
            self.doneCurrent()
            return None
        finally:
            self.doneCurrent()
        return elev, glow

    # ---- GL setup ----
    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glClearColor(0.016, 0.018, 0.030, 1.0)
        maj = int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION))
        mino = int(GL.glGetIntegerv(GL.GL_MINOR_VERSION))
        self.has_compute = (maj, mino) >= (4, 3)

        self.prog = self._make_program(PLANET_VS, PLANET_FS)
        self.water_prog = self._make_program(WATER_VS, WATER_FS)
        self.tree_prog = self._make_program(TREE_VS, TREE_FS)
        self.ring_prog = self._make_program(RING_VS, RING_FS)
        self.cloud_prog = self._make_program(CLOUD_VS, CLOUD_FS)
        self.cloudvol_prog = self._make_program(CLOUD_VS, CLOUDVOL_FS)
        self.atmo_prog = self._make_program(ATMOQ_VS, ATMOQ_FS)
        self.add_prog = self._make_program(QUAD_VS, ADD_FS)
        self.god_prog = self._make_program(QUAD_VS, GODRAY_FS)
        self.sunbody_prog = self._make_program(SUNBODY_VS, SUNBODY_FS)
        self.sun_prog = self._make_program(SUN_VS, SUN_FS)
        self.shell_prog = self._make_program(ATMOSHELL_VS, ATMOSHELL_FS)
        self.precip_prog = self._make_program(PRECIP_VS, PRECIP_FS)
        self.star_prog = self._make_program(STAR_VS, STAR_FS)
        self.line_prog = self._make_program(LINE_VS, LINE_FS)
        self.mark_prog = self._make_program(MARK_VS, MARK_FS)
        self.bright_prog = self._make_program(QUAD_VS, BRIGHT_FS)
        self.blur_prog = self._make_program(QUAD_VS, BLUR_FS)
        self.comp_prog = self._make_program(QUAD_VS, COMPOSITE_FS)

        q = np.array([-1, -1, 1, -1, 1, 1, -1, 1], np.float32)
        self._quad_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._quad_vao)
        qvbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, qvbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, q.nbytes, q, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)
        self._quad_vbo = qvbo

        self._corona_mesh = self._mesh(
            [(0, q.reshape(-1, 2))], None, GL.GL_TRIANGLE_FAN)
        self._marker_mesh = self._mesh([(0, np.zeros((1, 3), np.float32))],
                                       None, GL.GL_POINTS)
        sv, sn, si = build_icosphere(3)
        self._sun_mesh = self._mesh([(0, sv.astype(np.float32)),
                                     (1, sn.astype(np.float32))], si)
        self._shell_mesh = self._mesh([(0, build_icosphere(4)[0].astype(np.float32))],
                                      None)
        self._water_mesh = self._mesh([(0, build_icosphere(6)[0].astype(np.float32))],
                                      None)

        rng = np.random.default_rng(7)
        n = 2200
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        bn = np.array([0.32, 1.0, 0.18]); bn /= np.linalg.norm(bn)
        m = n // 2
        dp = d[:m] - (d[:m] @ bn)[:, None] * bn * 0.88
        dp += bn * rng.normal(0.0, 0.10, (m, 1))
        d[:m] = dp / np.maximum(np.linalg.norm(dp, axis=1, keepdims=True), 1e-9)
        pos = (d * (30.0 + 12.0 * rng.random(n))).astype(np.float32)
        size = (1.0 + 2.8 * rng.random(n) ** 2).astype(np.float32).reshape(-1, 1)
        bright = (0.18 + 0.85 * rng.random(n) ** 3).astype(np.float32).reshape(-1, 1)
        t = rng.random(n)[:, None]
        scol = np.clip((1.0, 0.78, 0.55) * (1.0 - t) + (0.72, 0.82, 1.0) * t
                       + rng.normal(0, 0.06, (n, 3)), 0.2, 1.0).astype(np.float32)
        self._star_mesh = self._mesh(
            [(0, pos), (1, size), (2, bright), (3, scol)], None, GL.GL_POINTS)

        segs = 220
        ang = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
        vin = np.stack([np.cos(ang), np.zeros(segs), np.sin(ang)], 1) * RING_INNER
        vout = np.stack([np.cos(ang), np.zeros(segs), np.sin(ang)], 1) * RING_OUTER
        rv = np.vstack([vin, vout]).astype(np.float32)
        idx = []
        for i in range(segs):
            j = (i + 1) % segs
            idx += [i, segs + j, j, i, segs + i, segs + j]
        self._ring_mesh = self._mesh([(0, rv)], np.array(idx, np.uint32))

        self._orbit_meshes = []
        for dist, spd, incl, ph in MOONS:
            a = np.linspace(0.0, 2.0 * np.pi, 160)
            pts = np.stack([dist * np.cos(a), dist * np.sin(a) * incl,
                            dist * np.sin(a)], 1).astype(np.float32)
            self._orbit_meshes.append(self._mesh([(0, pts)], None,
                                                 GL.GL_LINE_STRIP))

        self._moon_meshes = []
        for seed, scale, impacts, tint in _MOON_SPECS:
            mp = PlanetParams(seed=seed, subdiv=5, adaptive=0,
                              continent_scale=1.2, sea_level=-0.35,
                              mountain_height=0.04, warp=0.15, detail=0.7,
                              impacts=impacts, volcanoes=2, style="moon")
            mv, mf = build_icosphere(5)
            mdata = generate_planet(mp, mv, mf, scale=scale)
            mdata["colors"] = np.clip(
                mdata["colors"] * np.array(tint, np.float32), 0.0, 1.0)
            self._moon_meshes.append(self._mesh(
                [(0, mdata["positions"]), (1, mdata["normals"]),
                 (2, mdata["colors"]), (3, mdata["water"]), (4, mdata["emis"]),
                 (5, mdata["h"]), (6, mdata["ao"]), (7, mdata["rough"])],
                mdata["indices"]))

        # precipitation particle pool
        prng = np.random.default_rng(99)
        self._precip_pos = prng.uniform(-1.5, 1.5, (600, 3)).astype(np.float32)
        self._precip_mesh = self._mesh([(0, self._precip_pos)], None, GL.GL_POINTS)
        self._precip_vbo = self._precip_mesh.buffers[0]

        self._detail_tex = self._make_detail_tex()
        if self.planet is not None:
            self._build_planet_meshes(self.planet)

    def _make_program(self, vs, fs):
        prog = QOpenGLShaderProgram(self)
        if not (prog.addShaderFromSourceCode(QOpenGLShader.Vertex, vs)
                and prog.addShaderFromSourceCode(QOpenGLShader.Fragment, fs)
                and prog.link()):
            raise RuntimeError("Shader error:\n" + prog.log())
        return prog

    def _atmo_program(self):
        prog = self._atmo_progs.get(self.atmo_quality)
        if prog is None:
            pri, sec = _QUALITY[self.atmo_quality]
            fs = ATMOQ_FS.replace(
                "const float HR", f"#define PRI {pri}\n#define SEC {sec}\nconst float HR", 1)
            prog = self._make_program(ATMOQ_VS, fs)
            self._atmo_progs[self.atmo_quality] = prog
        return prog

    def _set_aniso(self, target):
        try:
            maxa = GL.glGetFloatv(GL.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            if maxa and maxa > 1.0:
                GL.glTexParameterf(target, GL.GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                   min(8.0, float(maxa)))
        except Exception:
            pass

    def _make_detail_tex(self):
        res = 256
        rng = np.random.default_rng(12345)
        W = np.fft.fft2(rng.standard_normal((res, res)))
        fy = np.fft.fftfreq(res)[:, None]
        fx = np.fft.fftfreq(res)[None, :]
        rad = np.sqrt(fx * fx + fy * fy)
        rad[0, 0] = 1.0
        H = W / rad ** 2.0
        H[0, 0] = 0.0
        h = np.fft.ifft2(H).real
        h -= h.mean()
        h /= max(h.std(), 1e-9)
        gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
        gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
        n = np.dstack([-gx * 3.0, -gy * 3.0, np.ones_like(h)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        rgb = np.ascontiguousarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, res, res, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        self._set_aniso(GL.GL_TEXTURE_2D)
        return tex

    def _mesh(self, attribs, indices=None, mode=GL.GL_TRIANGLES):
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        bufs = []
        for loc, arr in attribs:
            vbo = GL.glGenBuffers(1)
            bufs.append(vbo)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(loc, arr.shape[1], GL.GL_FLOAT,
                                     GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(loc)
        if indices is not None:
            ebo = GL.glGenBuffers(1)
            bufs.append(ebo)
            indices = np.ascontiguousarray(indices.astype(np.uint32))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                            indices, GL.GL_STATIC_DRAW)
            count = int(indices.size)
        else:
            count = int(attribs[0][1].shape[0])
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        return Mesh(vao, bufs, count, mode)

    @staticmethod
    def _delete_mesh(mesh):
        if mesh is None:
            return
        GL.glDeleteVertexArrays(1, [mesh.vao])
        GL.glDeleteBuffers(len(mesh.buffers), mesh.buffers)

    def _bake_clouds(self, seed):
        if self._cloud_tex is not None:
            GL.glDeleteTextures([self._cloud_tex])
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, tex)
        N = 64
        cn = SimplexNoise3D(seed * 3 + 911)
        mn = SimplexNoise3D(seed * 3 + 912)
        s = (np.arange(N) + 0.5) / N * 2.0 - 1.0
        S, T = np.meshgrid(s, s)
        for i, (maj, uax, vax) in enumerate(_CUBE_FACES):
            dx = maj[0] + S * uax[0] + T * vax[0]
            dy = maj[1] + S * uax[1] + T * vax[1]
            dz = maj[2] + S * uax[2] + T * vax[2]
            Ln = np.sqrt(dx * dx + dy * dy + dz * dz)
            dxn, dyn, dzn = dx / Ln, dy / Ln, dz / Ln
            nv = fbm(cn, dxn * 3.5, dyn * 3.5, dzn * 3.5, 5)
            dens = np.clip(0.5 + 0.6 * nv, 0.0, 1.0) ** 1.3
            t = 1.0 - np.abs(dyn) ** 1.4
            mp = 0.5 + 0.5 * fbm(mn, dxn * 3.1 + 4.2, dyn * 3.1 + 8.8,
                                 dzn * 3.1 + 1.9, 3)
            band = smoothstep(0.05, 0.30, t) * (1.0 - smoothstep(0.85, 1.0, t))
            itcz = np.exp(-(dyn * dyn) / 0.02)
            dens *= 0.25 + 0.85 * np.clip(0.35 * itcz + band * (0.4 + 0.9 * mp),
                                          0.0, 1.3)
            data = np.ascontiguousarray(
                (np.clip(dens, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(N, N, 1))
            GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_R8,
                            N, N, 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S,
                           GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T,
                           GL.GL_CLAMP_TO_EDGE)
        self._set_aniso(GL.GL_TEXTURE_CUBE_MAP)
        self._cloud_tex = tex
        # cyclone seeds
        rng = np.random.default_rng(seed * 13 + 71)
        cyc = []
        for i in range(6):
            c = rng.normal(size=3)
            c[1] *= 0.5
            c /= np.linalg.norm(c)
            if abs(c[1]) > 0.55:
                c[1] *= 0.3
                c /= np.linalg.norm(c)
            r = rng.uniform(0.05, 0.15)
            sign = 1.0 if i % 2 == 0 else -1.0
            cyc.append(QVector4D(c[0], c[1], c[2], sign * r))
        self._cyclones = cyc

    def _build_planet_meshes(self, data):
        self._delete_mesh(self._planet_mesh)
        self._planet_mesh = self._mesh(
            [(0, data["positions"]), (1, data["normals"]), (2, data["colors"]),
             (3, data["water"]), (4, data["emis"]), (5, data["h"]),
             (6, data["ao"]), (7, data["rough"])],
            data["indices"])
        d = data["positions"].astype(np.float64)
        self._dirs = (d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True),
                                     1e-12)).astype(np.float32)
        if self._tree_mesh is not None:
            self._delete_mesh(self._tree_mesh)
            self._tree_mesh = None
            self._tree_count = 0
        if data.get("trees") is not None:
            pos, nrm, idx = _tree_geometry()
            trees = data["trees"]
            vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(vao)
            v0 = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, v0)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, pos.nbytes, pos, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)
            v1 = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, v1)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, nrm.nbytes, nrm, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(1)
            e = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, e)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx,
                            GL.GL_STATIC_DRAW)
            vi = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vi)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, trees.nbytes, trees,
                            GL.GL_STATIC_DRAW)
            stride = 8 * 4
            for loc, off, sz in ((2, 0, 3), (3, 12, 1), (4, 16, 1), (5, 20, 3)):
                GL.glVertexAttribPointer(loc, sz, GL.GL_FLOAT, GL.GL_FALSE,
                                         stride, ctypes.c_void_p(off))
                GL.glEnableVertexAttribArray(loc)
                GL.glVertexAttribDivisor(loc, 1)
            GL.glBindVertexArray(0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            self._tree_mesh = Mesh(vao, [v0, v1, e, vi], int(idx.size),
                                   GL.GL_TRIANGLES, inst=int(trees.shape[0]))
            self._tree_count = self._tree_mesh.inst
        if self._cloud_seed != data["seed"]:
            self._cloud_seed = data["seed"]
            self._bake_clouds(data["seed"])
            self._delete_mesh(self._cloud_mesh)
            cv, cf = build_icosphere(5)
            self._cloud_mesh = self._mesh([(0, cv.astype(np.float32))], cf)

    # ---- FBOs ----
    def _tex2d(self, w, h, hdr):
        t = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0,
                        GL.GL_RGBA16F if hdr else GL.GL_RGBA8, w, h, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        return t

    def _fbo_with(self, tex, w, h, depth=False):
        fb = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb)
        rb = None
        if tex is not None:
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, tex, 0)
        if depth:
            rb = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24,
                                     w, h)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                         GL.GL_RENDERBUFFER, rb)
        ok = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        return fb, rb, ok

    def _ensure_fbo(self):
        dpr = self.devicePixelRatioF()
        fw = max(1, int(self.width() * dpr))
        fh = max(1, int(self.height() * dpr))
        if self._fbo_size == (fw, fh):
            return
        self._destroy_fbo()
        self._fbo_size = (fw, fh)
        ssw = max(1, int(fw * SS_FACTOR))
        ssh = max(1, int(fh * SS_FACTOR))
        self._ss = (ssw, ssh)
        self._half = (max(1, ssw // 2), max(1, ssh // 2))
        self._hdr = True
        self._scene_tex = self._tex2d(ssw, ssh, True)
        self._fbo_scene, self._depth_rb, ok = self._fbo_with(
            self._scene_tex, ssw, ssh, depth=True)
        if not ok:
            self._hdr = False
            GL.glDeleteTextures([self._scene_tex])
            GL.glDeleteFramebuffers(1, [self._fbo_scene])
            GL.glDeleteRenderbuffers(1, [self._depth_rb])
            self._depth_rb = None
            self._scene_tex = self._tex2d(ssw, ssh, False)
            self._fbo_scene, self._depth_rb, ok = self._fbo_with(
                self._scene_tex, ssw, ssh, depth=True)
        self._copy_tex = self._tex2d(ssw, ssh, self._hdr)
        self._fbo_copy, _, _ = self._fbo_with(self._copy_tex, ssw, ssh)
        self._tex_a = self._tex2d(*self._half, self._hdr)
        self._tex_b = self._tex2d(*self._half, self._hdr)
        self._fbo_a, _, _ = self._fbo_with(self._tex_a, *self._half)
        self._fbo_b, _, _ = self._fbo_with(self._tex_b, *self._half)
        self._atmo_tex = self._tex2d(*self._half, self._hdr)
        self._fbo_atmo, _, _ = self._fbo_with(self._atmo_tex, *self._half)
        self._fbo_ok = ok
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _destroy_fbo(self):
        for name in ("_fbo_scene", "_fbo_a", "_fbo_b", "_fbo_atmo", "_fbo_copy"):
            fb = getattr(self, name, None)
            if fb is not None:
                GL.glDeleteFramebuffers(1, [fb])
                setattr(self, name, None)
        for name in ("_scene_tex", "_tex_a", "_tex_b", "_atmo_tex", "_copy_tex"):
            t = getattr(self, name, None)
            if t is not None:
                GL.glDeleteTextures([t])
                setattr(self, name, None)
        if self._depth_rb is not None:
            GL.glDeleteRenderbuffers(1, [self._depth_rb])
            self._depth_rb = None
        self._fbo_ok = False
        self._fbo_size = None

    def _draw_quad(self):
        GL.glBindVertexArray(self._quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        GL.glBindVertexArray(0)

    def _post_process(self):
        fw, fh = self._fbo_size
        hw, hh = self._half
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_a)
        GL.glViewport(0, 0, hw, hh)
        self.bright_prog.bind()
        self.bright_prog.setUniformValue("uTex", 0)
        self.bright_prog.setUniformValue("uThresh", 1.05 if self._hdr else 0.70)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._scene_tex)
        self._draw_quad()
        self.bright_prog.release()
        self.blur_prog.bind()
        self.blur_prog.setUniformValue("uTex", 0)
        for _ in range(2):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_b)
            self.blur_prog.setUniformValue("uStep", 1.0 / hw, 0.0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_a)
            self._draw_quad()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_a)
            self.blur_prog.setUniformValue("uStep", 0.0, 1.0 / hh)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_b)
            self._draw_quad()
        self.blur_prog.release()
        gstr = 0.0
        if self._sun_uv[0] > -5.0 and self.god_strength > 0.01:
            gstr = float(self.god_strength * self._sun_occl)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_b)
            GL.glViewport(0, 0, hw, hh)
            self.god_prog.bind()
            self.god_prog.setUniformValue("uTex", 0)
            self.god_prog.setUniformValue("uSunUV", self._sun_uv[0], self._sun_uv[1])
            self.god_prog.setUniformValue("uOccl", float(self._sun_occl))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_a)
            self._draw_quad()
            self.god_prog.release()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, fw, fh)
        self.comp_prog.bind()
        self.comp_prog.setUniformValue("uScene", 0)
        self.comp_prog.setUniformValue("uBloom", 1)
        self.comp_prog.setUniformValue("uGod", 2)
        self.comp_prog.setUniformValue("uStrength", float(self.bloom))
        self.comp_prog.setUniformValue("uGStr", gstr)
        self.comp_prog.setUniformValue("uExposure", EXPOSURE)
        self.comp_prog.setUniformValue("uSunUV", self._sun_uv[0], self._sun_uv[1])
        self.comp_prog.setUniformValue("uOccl", float(self._sun_occl))
        self.comp_prog.setUniformValue("uAspect", fw / max(1.0, float(fh)))
        self.comp_prog.setUniformValue("uFlare", float(self.flare))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._scene_tex)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_a)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_b)
        self._draw_quad()
        self.comp_prog.release()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _light_vec(self):
        az, el = math.radians(self.sun_az), math.radians(self.sun_el)
        return QVector3D(math.cos(el) * math.cos(az), math.sin(el),
                         math.cos(el) * math.sin(az))

    def _moon_models(self):
        out = []
        for i in range(int(round(self.moon_count))):
            dist, spd, incl, ph = MOONS[i % len(MOONS)]
            scale = _MOON_SPECS[i % len(_MOON_SPECS)][1]
            a_deg = ph + self.moon_angle * (spd / 0.010)
            a = math.radians(a_deg)
            m = QMatrix4x4()
            m.translate(dist * math.cos(a), dist * math.sin(a) * incl,
                        dist * math.sin(a))
            m.rotate(a_deg, 0.0, 1.0, 0.0)
            out.append((m, m.map(QVector3D(0, 0, 0)), scale))
        return out

    def _set_cyc_uniforms(self, prog):
        if self._cyclones and self.show_clouds:
            prog.setUniformValueArray("uCyc", self._cyclones)
            prog.setUniformValue("uCycN", float(len(self._cyclones)))
        else:
            prog.setUniformValue("uCycN", 0.0)
        prog.setUniformValue("uCycT", float(self.time_s * 0.35))

    def _draw_scene(self, vp, eye, light):
        GL.glDepthMask(GL.GL_FALSE)
        self.star_prog.bind()
        self.star_prog.setUniformValue("uVP", vp)
        self.star_prog.setUniformValue("uTime", self.time_s)
        GL.glBindVertexArray(self._star_mesh.vao)
        GL.glDrawArrays(self._star_mesh.mode, 0, self._star_mesh.count)
        GL.glBindVertexArray(0)
        self.star_prog.release()
        # corona behind the sun body
        self.sun_prog.bind()
        self.sun_prog.setUniformValue("uVP", vp)
        self.sun_prog.setUniformValue("uCenter", light * 55.0)
        self.sun_prog.setUniformValue("uEye", eye)
        self.sun_prog.setUniformValue("uSize", 4.4)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBindVertexArray(self._corona_mesh.vao)
        GL.glDrawArrays(self._corona_mesh.mode, 0, self._corona_mesh.count)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
        self.sun_prog.release()
        # sun body (granulated, HDR) — occluded by the planet via depth
        smodel = QMatrix4x4()
        smodel.translate(light.x() * 55.0, light.y() * 55.0, light.z() * 55.0)
        smodel.scale(2.6)
        sp = self.sunbody_prog
        sp.bind()
        sp.setUniformValue("uMVP", vp * smodel)
        sp.setUniformValue("uModel", smodel)
        sp.setUniformValue("uEye", eye)
        sp.setUniformValue("uTime", self.time_s)
        GL.glBindVertexArray(self._sun_mesh.vao)
        GL.glDrawElements(self._sun_mesh.mode, self._sun_mesh.count,
                          GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        sp.release()
        GL.glDepthMask(GL.GL_TRUE)
        if self._planet_mesh is None:
            return False

        planet = self.planet
        style = STYLES.get(planet["style"], STYLES["terran"])
        if self.terraform > 0.0 and planet["style"] == "mars":
            T = STYLES["terran"]
            style = dict(style)
            for k in ("betaR", "sky", "cloud"):
                style[k] = tuple(a + (b - a) * self.terraform
                                 for a, b in zip(style[k], T[k]))
        draw_rings = (self.ring_opacity > 0.02 and not self.wireframe
                      and not self._sky_mode)
        model = QMatrix4x4()
        model.rotate(self.tilt, 0.0, 0.0, 1.0)
        model.rotate(self.spin, 0.0, 1.0, 0.0)
        inv = QMatrix4x4()
        inv.rotate(-self.spin, 0.0, 1.0, 0.0)
        inv.rotate(-self.tilt, 0.0, 0.0, 1.0)
        if not self._sky_mode:
            self._model_inv = inv
        ll = inv.map(light)
        if ll.length() > 1e-6:
            ll = QVector3D(ll.x() / ll.length(), ll.y() / ll.length(),
                           ll.z() / ll.length())
        do_clouds = (self.show_clouds and self._cloud_mesh is not None
                     and not self.wireframe and style["cloud_max"] > 0.0
                     and not self._sky_mode)
        edge = 1.0 - 0.9 * self.cloud_cover
        moons = self._moon_models() if self.show_moon else []
        mvec = []
        for _, wpos, r0 in moons[:3]:
            lo = inv.map(wpos)
            mvec.append(QVector4D(lo.x(), lo.y(), lo.z(), r0))
        while len(mvec) < 3:
            mvec.append(QVector4D(0, 0, 0, 0))

        p = self.prog
        p.bind()
        p.setUniformValue("uMVP", vp * model)
        p.setUniformValue("uModel", model)
        p.setUniformValue("uLightDir", light)
        p.setUniformValue("uLLight", ll)
        p.setUniformValue("uEye", eye)
        p.setUniformValue("uGlowColor",
                          QVector3D(*style["glow"]) if self.show_lights
                          else QVector3D(0, 0, 0))
        p.setUniformValue("uSkyTint", QVector3D(*style.get("sky", (0, 0, 0))))
        p.setUniformValue("uNight",
                          float(style["night"]) if self.show_lights else 0.0)
        p.setUniformValue("uTime", self.time_s)
        p.setUniformValue("uRingShade", 1.0 if draw_rings else 0.0)
        p.setUniformValue("uCloudTex", 0)
        p.setUniformValue("uEdge", float(edge))
        p.setUniformValue("uCloudRot", math.radians(self.cloud_drift))
        p.setUniformValue("uCloudShadow", 1.0 if do_clouds else 0.0)
        p.setUniformValue("uRelTex", 2)
        p.setUniformValue("uRelStrength",
                          0.0 if (planet["style"] == "gas" or self._sky_mode)
                          else float(self.relief_strength))
        p.setUniformValue("uMoonPos", mvec)
        p.setUniformValue("uMoonN", float(len(moons[:3]) if not self._sky_mode else 0))
        if not self._sky_mode:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self._cloud_tex)
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._detail_tex)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK,
                             GL.GL_LINE if self.wireframe else GL.GL_FILL)
            GL.glBindVertexArray(self._planet_mesh.vao)
            GL.glDrawElements(self._planet_mesh.mode, self._planet_mesh.count,
                              GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            # water plane with screen-space refraction
            if (self._fbo_ok and not self.wireframe and self._water_mesh
                    and planet["stats"].get("sea")):
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._fbo_scene)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._fbo_copy)
                GL.glBlitFramebuffer(0, 0, self._ss[0], self._ss[1],
                                     0, 0, self._ss[0], self._ss[1],
                                     GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_scene)
                self._set_cyc_uniforms(self.water_prog)
                wp = self.water_prog
                wp.bind()
                wp.setUniformValue("uMVP", vp * model)
                wp.setUniformValue("uModel", model)
                wp.setUniformValue("uScene", 3)
                wp.setUniformValue("uRes", self._ss[0], self._ss[1])
                wp.setUniformValue("uEye", eye)
                wp.setUniformValue("uLight", light)
                wp.setUniformValue("uSkyTint", QVector3D(*style["sky"]))
                wp.setUniformValue("uDeep", QVector3D(*style["deep"]))
                wp.setUniformValue("uGlow",
                                   QVector3D(*style["water_glow"])
                                   if self.show_lights else QVector3D(0, 0, 0))
                wp.setUniformValue("uTime", self.time_s)
                wp.setUniformValue("uCloudTex", 0)
                wp.setUniformValue("uEdge", float(edge))
                wp.setUniformValue("uCloudRot", math.radians(self.cloud_drift))
                wp.setUniformValue("uCloudShadow", 1.0 if do_clouds else 0.0)
                GL.glActiveTexture(GL.GL_TEXTURE3)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._copy_tex)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self._cloud_tex)
                GL.glDisable(GL.GL_CULL_FACE)
                GL.glBindVertexArray(self._water_mesh.vao)
                GL.glDrawElements(self._water_mesh.mode, self._water_mesh.count,
                                  GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)
                GL.glEnable(GL.GL_CULL_FACE)
                wp.release()
            if (self.show_veg and self._tree_mesh is not None
                    and not self.wireframe):
                fade = max(0.0, min(1.0, (6.5 - self.dist) / 2.5))
                k = int(self._tree_count * self.veg_density)
                if k > 0 and fade > 0.02:
                    tp = self.tree_prog
                    tp.bind()
                    tp.setUniformValue("uMVP", vp * model)
                    tp.setUniformValue("uModel", model)
                    tp.setUniformValue("uLightDir", light)
                    tp.setUniformValue("uFade", fade)
                    GL.glBindVertexArray(self._tree_mesh.vao)
                    GL.glDrawElementsInstanced(
                        GL.GL_TRIANGLES, self._tree_mesh.count,
                        GL.GL_UNSIGNED_INT, None, k)
                    GL.glBindVertexArray(0)
                    tp.release()
            if self._marker_on:
                self.mark_prog.bind()
                self.mark_prog.setUniformValue("uVP", vp)
                self.mark_prog.setUniformValue("uModel", model)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._marker_mesh.buffers[0])
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0,
                                   self._marker_pos.nbytes, self._marker_pos)
                GL.glBindVertexArray(self._marker_mesh.vao)
                GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                GL.glBindVertexArray(0)
                self.mark_prog.release()
        for i, (mmodel, _, _) in enumerate(moons):
            mesh = self._moon_meshes[i % len(self._moon_meshes)]
            p.setUniformValue("uMVP", vp * mmodel)
            p.setUniformValue("uModel", mmodel)
            p.setUniformValue("uMoonN", 0.0)
            p.setUniformValue("uCloudShadow", 0.0)
            p.setUniformValue("uRingShade", 0.0)
            GL.glBindVertexArray(mesh.vao)
            GL.glDrawElements(mesh.mode, mesh.count, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
        p.release()

        if self.show_orbits and self._orbit_meshes and moons and not self._sky_mode:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            self.line_prog.bind()
            self.line_prog.setUniformValue("uVP", vp)
            self.line_prog.setUniformValue("uColor", 0.55, 0.65, 0.90, 0.22)
            for om in self._orbit_meshes[:len(moons)]:
                GL.glBindVertexArray(om.vao)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, om.count)
            GL.glBindVertexArray(0)
            self.line_prog.release()
            GL.glDisable(GL.GL_BLEND)
            GL.glDepthMask(GL.GL_TRUE)

        if self._sky_mode:
            return False

        # binary companion planet
        if self.show_companion and not self.wireframe:
            ca = math.radians(140.0 + self.moon_angle * 0.35)
            cd = 8.6
            cmodel = QMatrix4x4()
            cmodel.translate(cd * math.cos(ca), cd * math.sin(ca) * 0.12,
                             cd * math.sin(ca))
            cmodel.rotate(self.spin * 0.6 + 90.0, 0.0, 1.0, 0.0)
            cmodel.scale(0.55)
            p.bind()
            p.setUniformValue("uMVP", vp * cmodel)
            p.setUniformValue("uModel", cmodel)
            p.setUniformValue("uMoonN", 0.0)
            p.setUniformValue("uCloudShadow", 0.0)
            p.setUniformValue("uRingShade", 0.0)
            GL.glBindVertexArray(self._planet_mesh.vao)
            GL.glDrawElements(self._planet_mesh.mode, self._planet_mesh.count,
                              GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
            p.release()
            if self.show_moon_atmo:
                ashell = QMatrix4x4(cmodel)
                GL.glDepthMask(GL.GL_FALSE)
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
                shl = self.shell_prog
                shl.bind()
                shl.setUniformValue("uVP", vp)
                shl.setUniformValue("uModel", ashell)
                shl.setUniformValue("uEye", eye)
                shl.setUniformValue("uLight", light)
                shl.setUniformValue("uCol", QVector3D(*style["sky"]) * 1.2)
                GL.glBindVertexArray(self._shell_mesh.vao)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._shell_mesh.count)
                GL.glBindVertexArray(0)
                shl.release()
                GL.glDisable(GL.GL_BLEND)
                GL.glDepthMask(GL.GL_TRUE)
            if draw_rings:
                self._draw_ring(vp, cmodel, self.ring_opacity * 0.8, light,
                                planet["seed"])

        # moon rings + atmospheres
        if self.show_moon_rings and moons:
            for i, (mmodel, _, mr) in enumerate(moons):
                s = mr * 2.6 / RING_OUTER
                rmodel = QMatrix4x4(mmodel)
                rmodel.scale(s)
                self._draw_ring(vp, rmodel, 0.5, light, planet["seed"] + i)
        if self.show_moon_atmo and moons:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
            shl = self.shell_prog
            shl.bind()
            shl.setUniformValue("uVP", vp)
            shl.setUniformValue("uEye", eye)
            shl.setUniformValue("uLight", light)
            shl.setUniformValue("uCol", QVector3D(0.45, 0.65, 1.0))
            mmodel, _, mr = moons[0]
            am = QMatrix4x4(mmodel)
            am.scale(1.14)
            shl.setUniformValue("uModel", am)
            GL.glBindVertexArray(self._shell_mesh.vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._shell_mesh.count)
            GL.glBindVertexArray(0)
            shl.release()
            GL.glDisable(GL.GL_BLEND)
            GL.glDepthMask(GL.GL_TRUE)

        if draw_rings or do_clouds:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
        if draw_rings:
            rmodel = QMatrix4x4()
            rmodel.rotate(self.tilt, 0.0, 0.0, 1.0)
            self._draw_ring(vp, rmodel, self.ring_opacity, light, planet["seed"])
        if do_clouds:
            cmodel = QMatrix4x4()
            cmodel.rotate(self.tilt, 0.0, 0.0, 1.0)
            cmodel.rotate(self.spin + self.cloud_drift, 0.0, 1.0, 0.0)
            use_vol = self.atmo_quality == "High"
            cp = self.cloudvol_prog if use_vol else self.cloud_prog
            cp.bind()
            cp.setUniformValue("uMVP", vp * cmodel)
            cp.setUniformValue("uModel", cmodel)
            cp.setUniformValue("uEye", eye)
            cp.setUniformValue("uCloudTex", 0)
            cp.setUniformValue("uLight", light)
            cp.setUniformValue("uLightDir", light)
            cp.setUniformValue("uCloudColor", QVector3D(*style["cloud"]))
            cp.setUniformValue("uEdge", float(edge))
            cp.setUniformValue("uDrift", math.radians(self.cloud_drift))
            self._set_cyc_uniforms(cp)
            if use_vol:
                cp.setUniformValue("uR0", float(planet["cloud_radius"] - 0.02))
                cp.setUniformValue("uR1", float(planet["cloud_radius"] + 0.05))
            else:
                cp.setUniformValue("uRadius", float(planet["cloud_radius"]))
                cp.setUniformValue("uCloudMax", float(style["cloud_max"]))
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self._cloud_tex)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glBindVertexArray(self._cloud_mesh.vao)
            GL.glDrawElements(self._cloud_mesh.mode, self._cloud_mesh.count,
                              GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
            cp.release()
        if draw_rings or do_clouds:
            GL.glDisable(GL.GL_BLEND)
            GL.glDepthMask(GL.GL_TRUE)

        # precipitation near the camera
        if (self.show_weather and self.show_clouds and self._precip_mesh
                and self.dist < 3.2 and planet["style"] in ("terran", "ice")
                and not self.wireframe):
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
            pp = self.precip_prog
            pp.bind()
            pp.setUniformValue("uVP", vp)
            pp.setUniformValue("uCenter", eye)
            snow = planet["style"] == "ice"
            pp.setUniformValue("uSize", 3.0 if snow else 2.0)
            pp.setUniformValue("uColor",
                               QVector3D(1.0, 1.0, 1.0) if snow
                               else QVector3D(0.55, 0.70, 1.0))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._precip_vbo)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, self._precip_pos.nbytes,
                               self._precip_pos)
            GL.glBindVertexArray(self._precip_mesh.vao)
            GL.glDrawArrays(GL.GL_POINTS, 0, self._precip_pos.shape[0])
            GL.glBindVertexArray(0)
            pp.release()
            GL.glDisable(GL.GL_BLEND)
            GL.glDepthMask(GL.GL_TRUE)

        return (self.show_atmo and style["atmo"] and not self.wireframe)

    def _draw_ring(self, vp, model, opacity, light, seed):
        GL.glDisable(GL.GL_CULL_FACE)
        rp = self.ring_prog
        rp.bind()
        rp.setUniformValue("uVP", vp)
        rp.setUniformValue("uModel", model)
        rp.setUniformValue("uLightDir", light)
        rp.setUniformValue("uOpacity", float(opacity))
        rp.setUniformValue("uSeedPh", float((seed % 89) * 0.7))
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindVertexArray(self._ring_mesh.vao)
        GL.glDrawElements(self._ring_mesh.mode, self._ring_mesh.count,
                          GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        rp.release()
        GL.glEnable(GL.GL_CULL_FACE)

    def _render_atmosphere(self, vp):
        planet = self.planet
        style = STYLES.get(planet["style"], STYLES["terran"])
        if self.terraform > 0.0 and planet["style"] == "mars":
            T = STYLES["terran"]
            style = dict(style)
            style["betaR"] = tuple(a + (b - a) * self.terraform
                                   for a, b in zip(style["betaR"], T["betaR"]))
        inv_vp, _ = vp.inverted()
        t = math.radians(self.tilt)
        ap = self._atmo_program()
        ap.bind()
        ap.setUniformValue("uInvVP", inv_vp)
        ap.setUniformValue("uEye", self._eye)
        ap.setUniformValue("uLight", self._light_vec())
        ap.setUniformValue("uBetaR", QVector3D(*style["betaR"]))
        density = self.atmo_density
        if planet["style"] == "mars" and self.terraform > 0.0:
            density *= (1.0 + 1.3 * self.terraform)
        ap.setUniformValue("uDensity", float(density))
        ap.setUniformValue("uSunI", 18.0)
        ap.setUniformValue("uAtmoR", max(1.12, planet["cloud_radius"] + 0.08))
        ap.setUniformValue("uTime", self.time_s)
        ap.setUniformValue("uMulti", float(self.multi))
        aur = style.get("aurora")
        if aur and self.show_aurora:
            ap.setUniformValue("uAurora", 1.0)
            ap.setUniformValue("uAurColor", QVector3D(0.4, 1.0, 0.6))
        else:
            ap.setUniformValue("uAurora", 0.0)
            ap.setUniformValue("uAurColor", QVector3D(0, 0, 0))
        ap.setUniformValue("uAurAxis", QVector3D(-math.sin(t), math.cos(t), 0.0))
        ap.setUniformValue("uAurX", QVector3D(math.cos(t), math.sin(t), 0.0))
        ap.setUniformValue("uAurY", QVector3D(0.0, 0.0, 1.0))
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_CULL_FACE)
        if self._fbo_ok:
            hw, hh = self._half
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_atmo)
            GL.glViewport(0, 0, hw, hh)
            GL.glClearColor(0, 0, 0, 0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            self._draw_quad()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_scene)
            GL.glViewport(0, 0, self._ss[0], self._ss[1])
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
            self.add_prog.bind()
            self.add_prog.setUniformValue("uTex", 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._atmo_tex)
            self._draw_quad()
            self.add_prog.release()
            GL.glDisable(GL.GL_BLEND)
            GL.glClearColor(0.016, 0.018, 0.030, 1.0)
        else:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
            self._draw_quad()
            GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        ap.release()

    def paintGL(self):
        self._ensure_fbo()
        if self._fbo_ok:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_scene)
            GL.glViewport(0, 0, self._ss[0], self._ss[1])
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        proj = QMatrix4x4()
        proj.perspective(45.0, self.width() / max(1, self.height()), 0.1, 200.0)
        view = QMatrix4x4()
        view.rotate(self.pitch, 1.0, 0.0, 0.0)
        view.rotate(self.yaw, 0.0, 1.0, 0.0)
        view.translate(0.0, 0.0, -self.dist)
        vp = proj * view
        rm = QMatrix4x4()
        rm.rotate(-self.yaw, 0.0, 1.0, 0.0)
        rm.rotate(-self.pitch, 1.0, 0.0, 0.0)
        self._eye = rm.map(QVector3D(0.0, 0.0, self.dist))
        self._inv_vp, _ = vp.inverted()
        light = self._light_vec()
        # sun projection + analytic occlusion (for god rays + lens flare)
        self._sun_uv = (-10.0, -10.0)
        self._sun_occl = 0.0
        p4 = vp * QVector4D(light.x() * 55.0, light.y() * 55.0,
                            light.z() * 55.0, 1.0)
        if p4.w() > 0.001:
            u = p4.x() / p4.w() * 0.5 + 0.5
            v = p4.y() / p4.w() * 0.5 + 0.5
            if -0.15 <= u <= 1.15 and -0.15 <= v <= 1.15:
                eye_v = np.array([self._eye.x(), self._eye.y(), self._eye.z()])
                s_v = np.array([light.x(), light.y(), light.z()]) * 55.0
                dv = s_v - eye_v
                dist = np.linalg.norm(dv)
                occl = 1.0
                if dist > 1e-6:
                    dirv = dv / dist
                    tc = float(-np.dot(eye_v, dirv))
                    if tc > 0.0:
                        d2 = float(np.dot(eye_v, eye_v) - tc * tc)
                        if d2 < 1.0:
                            occl = float(np.clip(
                                (math.sqrt(max(d2, 0.0)) - 0.85) / 0.20, 0.0, 1.0))
                self._sun_uv = (u, v)
                self._sun_occl = occl
        do_atmo = self._draw_scene(vp, self._eye, light)
        if do_atmo:
            self._render_atmosphere(vp)
        if self._fbo_ok:
            self._post_process()

    def resizeGL(self, w, h):
        dpr = self.devicePixelRatioF()
        GL.glViewport(0, 0, int(w * dpr), int(h * dpr))

    # ---- picking ----
    def _ray(self, x, y):
        if self._inv_vp is None:
            return None
        nx = 2.0 * x / max(1, self.width()) - 1.0
        ny = 1.0 - 2.0 * y / max(1, self.height())
        pf = self._inv_vp.map(QVector4D(nx, ny, 1.0, 1.0))
        pn = self._inv_vp.map(QVector4D(nx, ny, -1.0, 1.0))
        f = np.array([pf.x() / pf.w(), pf.y() / pf.w(), pf.z() / pf.w()])
        nn = np.array([pn.x() / pn.w(), pn.y() / pn.w(), pn.z() / pn.w()])
        rd = f - nn
        n = np.linalg.norm(rd)
        if n < 1e-9:
            return None
        return np.array([self._eye.x(), self._eye.y(), self._eye.z()]), rd / n

    def _ray_hit(self, x, y):
        r = self._ray(x, y)
        if r is None:
            return None
        ro, rd = r
        b = ro @ rd
        c = ro @ ro - 1.0
        h = b * b - c
        if h < 0:
            return None
        t = -b - math.sqrt(h)
        if t < 0:
            return None
        return ro, rd, t

    def _local_dir(self, ro, rd, t):
        wp = ro + rd * t
        lp = self._model_inv.map(QVector3D(*wp))
        v = np.array([lp.x(), lp.y(), lp.z()])
        n = np.linalg.norm(v)
        return v / max(n, 1e-9)

    def _probe(self, pos):
        hit = self._ray_hit(pos.x(), pos.y())
        if hit is None or self.planet is None or self._dirs is None:
            return
        ro, rd, t = hit
        d = self._local_dir(ro, rd, t)
        lat = math.degrees(math.asin(np.clip(d[1], -1, 1)))
        lon = math.degrees(math.atan2(d[2], d[0]))
        dd = self._dirs - d.astype(np.float32)
        i = int(np.argmin(np.einsum("ij,ij->i", dd, dd)))
        self._marker_pos = self.planet["positions"][i].copy()
        self._marker_on = True
        self.update()
        elev = float(self.planet["h"][i, 0])
        self.probePicked.emit(dict(
            lat=lat, lon=lon, elev=elev,
            temp=float(self.planet["temp"][i]),
            moist=float(self.planet["moist"][i]),
            biome=BIOME_NAMES[int(self.planet["biome"][i])]))

    # ---- interaction ----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._last = e.position()
        elif e.button() == Qt.RightButton:
            self._probe(e.position())

    def mouseMoveEvent(self, e):
        if self._last is not None and (e.buttons() & Qt.LeftButton):
            d = e.position() - self._last
            self.yaw = (self.yaw + d.x() * 0.4) % 360.0
            self.pitch = max(-89.0, min(89.0, self.pitch + d.y() * 0.4))
            self._last = e.position()
            self.update()
        elif self._last is None and self.hover_cb and self.planet is not None:
            now = time.perf_counter()
            if now - self._hover_t > 0.06:
                self._hover_t = now
                hit = self._ray_hit(e.position().x(), e.position().y())
                if hit:
                    d = self._local_dir(*hit)
                    self.hover_cb(math.degrees(math.asin(np.clip(d[1], -1, 1))),
                                  math.degrees(math.atan2(d[2], d[0])))

    def mouseReleaseEvent(self, e):
        self._last = None

    def mouseDoubleClickEvent(self, e):
        self.yaw, self.pitch, self.dist = 25.0, 12.0, 3.4
        self._anim = None
        self.update()

    def wheelEvent(self, e):
        self.dist = max(1.7, min(14.0,
                                 self.dist * (1.0 - e.angleDelta().y() * 0.0008)))
        self.update()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class GenWorker(QThread):
    finished_gen = Signal(object)
    failed = Signal(str)

    def __init__(self, params, gpu_elev=None, gpu_glow=None, parent=None):
        super().__init__(parent)
        self.params = params
        self.gpu_elev = gpu_elev
        self.gpu_glow = gpu_glow

    def run(self):
        try:
            t0 = time.perf_counter()
            if self.gpu_elev is not None:
                verts, faces = get_icosphere(self.params.subdiv)
                data = generate_planet(self.params, verts, faces,
                                       elev0=self.gpu_elev, emis0=self.gpu_glow)
            else:
                data = generate_planet(self.params)
            data["_ms"] = (time.perf_counter() - t0) * 1000.0
            if data["n_vertices"] < 20000:
                data["_minimap"] = quick_map(data)
            self.finished_gen.emit(data)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class ExportWorker(QThread):
    finished_export = Signal(str, str)
    failed = Signal(str)

    def __init__(self, kind, path, data=None, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.path = path
        self.data = data

    def run(self):
        try:
            if self.kind == "obj":
                export_obj(self.path, self.data)
                self.finished_export.emit(self.kind, f"OBJ saved:\n{self.path}")
            elif self.kind == "glb":
                alb, nrm, _, _ = render_equirect_from_mesh(self.data, 2048, 1024)
                export_glb(self.path, self.data,
                           png_bytes_rgb(alb), png_bytes_rgb(nrm))
                self.finished_export.emit(
                    self.kind, "glTF binary saved (mesh + albedo + normal):\n"
                    + self.path)
            else:
                alb, nrm, h8, h16 = render_equirect_from_mesh(self.data, 2048, 1024)
                base = (self.path[:-4] if self.path.lower().endswith(".png")
                        else self.path)
                h, w = alb.shape[:2]
                saved = []
                for arr, suffix in ((alb, "albedo"), (nrm, "normal"),
                                    (h8, "height8")):
                    p = f"{base}_{suffix}.png"
                    buf = arr.tobytes()
                    img = (QImage(buf, w, h, 3 * w, QImage.Format_RGB888)
                           if arr.ndim == 3 else
                           QImage(buf, w, h, w, QImage.Format_Grayscale8))
                    img.save(p, "PNG")
                    saved.append(p)
                p16 = f"{base}_height16.png"
                write_png16(p16, h16)
                saved.append(p16)
                self.finished_export.emit(
                    self.kind, "Map pack saved (2048\u00d71024, incl 16-bit height):\n"
                    + "\n".join(saved))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

PRESETS = {
    "Terran":      dict(style="terran", adaptive=1, continent=1.5, sea=0.03, mountains=0.14, warp=0.40, detail=1.0, impacts=0, volcanoes=0, rain=0.00, climate=0.00, erosion=18, thermal=8, rivers=0.65, rings=0.0, moons=1),
    "Archipelago": dict(style="terran", adaptive=2, continent=3.4, sea=0.30, mountains=0.09, warp=0.62, detail=1.25, impacts=0, volcanoes=3, rain=0.12, climate=0.06, erosion=8, thermal=6, rivers=0.35, rings=0.0, moons=1),
    "Ocean":       dict(style="terran", adaptive=1, continent=1.2, sea=0.32, mountains=0.10, warp=0.50, detail=1.0, impacts=0, volcanoes=4, rain=0.15, climate=0.00, erosion=10, thermal=6, rivers=0.30, rings=0.0, moons=1),
    "Desert":      dict(style="terran", adaptive=1, continent=1.4, sea=-0.20, mountains=0.08, warp=0.30, detail=0.8, impacts=8, volcanoes=2, rain=-0.32, climate=0.10, erosion=25, thermal=20, rivers=0.05, rings=0.0, moons=0),
    "Ice":         dict(style="ice", adaptive=1, continent=1.6, sea=0.10, mountains=0.16, warp=0.35, detail=0.9, impacts=5, volcanoes=0, rain=0.00, climate=-0.22, erosion=12, thermal=10, rivers=0.35, rings=0.0, moons=1),
    "Glacial":     dict(style="ice", adaptive=2, continent=1.8, sea=0.05, mountains=0.20, warp=0.55, detail=1.0, impacts=3, volcanoes=0, rain=0.10, climate=-0.30, erosion=14, thermal=6, rivers=0.55, rings=0.0, moons=1),
    "Mars":        dict(style="mars", adaptive=1, continent=1.3, sea=-0.45, mountains=0.10, warp=0.45, detail=0.9, impacts=30, volcanoes=8, rain=0.00, climate=-0.05, erosion=35, thermal=16, rivers=0.18, rings=0.0, moons=2),
    "Moon":        dict(style="moon", adaptive=0, continent=1.0, sea=-0.35, mountains=0.05, warp=0.20, detail=0.6, impacts=95, volcanoes=2, rain=0.00, climate=0.00, erosion=0, thermal=0, rivers=0.0, rings=0.0, moons=0, clouds=False, atmo=False),
    "Lava":        dict(style="lava", adaptive=1, continent=1.6, sea=0.05, mountains=0.12, warp=0.50, detail=1.0, impacts=10, volcanoes=24, rain=0.00, climate=0.20, erosion=6, thermal=4, rivers=0.0, rings=0.0, moons=1),
    "Jupiter":     dict(style="gas", adaptive=0, continent=1.5, sea=0.0, mountains=0.0, warp=0.50, detail=1.0, impacts=0, volcanoes=0, rain=0.00, climate=0.00, erosion=0, thermal=0, rivers=0.0, rings=0.0, moons=3),
    "Saturn":      dict(style="gas", adaptive=0, continent=1.5, sea=0.0, mountains=0.0, warp=0.50, detail=1.0, impacts=0, volcanoes=0, rain=0.00, climate=0.00, erosion=0, thermal=0, rivers=0.0, rings=0.85, moons=2),
}


class MainWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("Planet Generator v8")
        self.resize(1440, 920)
        self._base_subdiv = 6
        self.setAcceptDrops(True)

        self.gl_widget = PlanetWidget()
        self._worker = None
        self._pending = False
        self._export_worker = None
        self._sliders = {}

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.gl_widget, 1)
        self.panel = self._build_panel()
        root.addWidget(self.panel)
        self.setCentralWidget(central)
        self._make_shortcuts()

        self.gl_widget.subdiv_change_cb = self._on_hires
        self.gl_widget.rotate_cb = lambda c: self.chk_rotate.setChecked(c)
        self.gl_widget.sun_ui_cb = self._sun_ui
        self.gl_widget.hover_cb = self._on_hover
        self.gl_widget.fps_cb = self._on_fps
        self.gl_widget.probePicked.connect(self._on_probe)

        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(120)
        self._regen_timer.timeout.connect(self._regen)

        self._live_sync()
        used_cli = False
        if args and (args.seed or args.style or args.subdiv or args.preset):
            self._apply_cli(args)
            used_cli = True
        elif os.path.exists(_CFG_PATH):
            try:
                with open(_CFG_PATH, encoding="utf-8") as fh:
                    self._config_apply(json.load(fh))
            except Exception:
                pass
        self._regen()
        self._used_cli = used_cli

    # ---- drag & drop: PNG with embedded config, or .json ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".json"):
                try:
                    with open(path, encoding="utf-8") as fh:
                        self._config_apply(json.load(fh))
                    self.stats.setText("config loaded from " + path)
                except Exception as ex:
                    QMessageBox.critical(self, "Config error", str(ex))
                return
            if path.lower().endswith(".png"):
                raw = read_png_text(path, "planetConfig")
                if raw:
                    try:
                        self._config_apply(json.loads(raw))
                        self.stats.setText("planet reproduced from " + path)
                    except Exception as ex:
                        QMessageBox.critical(self, "Config error", str(ex))
                else:
                    self.stats.setText("no embedded planet config in " + path)
                return

    # ---- panel ----
    def _build_panel(self):
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        g = QGroupBox("Dossier")
        dv = QVBoxLayout(g)
        self.name_lbl = QLabel("\u2014")
        self.name_lbl.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.meta_lbl = QLabel("\u2014")
        self.meta_lbl.setStyleSheet("color: #9aa0aa;")
        self.meta_lbl.setWordWrap(True)
        self.stat_lbl = QLabel("\u2014")
        self.stat_lbl.setWordWrap(True)
        self.map_lbl = QLabel()
        self.map_lbl.setFixedSize(356, 178)
        self.map_lbl.setAlignment(Qt.AlignCenter)
        self.map_lbl.setStyleSheet("background: #10121a; border: 1px solid #2a2e38;")
        dv.addWidget(self.name_lbl)
        dv.addWidget(self.meta_lbl)
        dv.addWidget(self.stat_lbl)
        dv.addWidget(self.map_lbl)
        v.addWidget(g)

        g = QGroupBox("Survey")
        sv = QVBoxLayout(g)
        self.hover_lbl = QLabel("cursor: \u2014")
        self.hover_lbl.setStyleSheet("color: #9aa0aa;")
        self.probe_lbl = QLabel("right-click the globe to survey")
        self.probe_lbl.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(QLabel("Lat"))
        self.lat_box = QDoubleSpinBox()
        self.lat_box.setRange(-90.0, 90.0)
        self.lat_box.setDecimals(1)
        row.addWidget(self.lat_box)
        row.addWidget(QLabel("Lon"))
        self.lon_box = QDoubleSpinBox()
        self.lon_box.setRange(-180.0, 180.0)
        self.lon_box.setDecimals(1)
        row.addWidget(self.lon_box)
        go = QPushButton("Go")
        go.clicked.connect(self._go_to)
        row.addWidget(go)
        sv.addWidget(self.hover_lbl)
        sv.addWidget(self.probe_lbl)
        sv.addLayout(row)
        v.addWidget(g)

        g = QGroupBox("World")
        gv = QVBoxLayout(g)
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed"))
        self.seed_box = QSpinBox()
        self.seed_box.setRange(1, 999999)
        self.seed_box.setValue(random.randint(1, 999999))
        self.seed_box.valueChanged.connect(self._schedule_regen)
        seed_row.addWidget(self.seed_box, stretch=1)
        rand_btn = QPushButton("Random")
        rand_btn.clicked.connect(
            lambda: self.seed_box.setValue(random.randint(1, 999999)))
        seed_row.addWidget(rand_btn)
        gv.addLayout(seed_row)
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution"))
        self.subdiv_box = QSpinBox()
        self.subdiv_box.setRange(2, 9)
        self.subdiv_box.setValue(6)
        self.subdiv_box.setToolTip(
            "Icosphere subdivision (GPU recommended for 8\u20139)\n"
            "7: 163,842   8: 655,362   9: 2,621,442 verts")
        self.subdiv_box.valueChanged.connect(self._subdiv_changed)
        res_row.addWidget(self.subdiv_box, stretch=1)
        gv.addLayout(res_row)
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("World type"))
        self.style_combo = QComboBox()
        for s in ("Terran", "Ice", "Mars", "Lava", "Moon", "Gas"):
            self.style_combo.addItem(s.capitalize())
        self.style_combo.setCurrentText("Terran")
        self.style_combo.currentTextChanged.connect(self._schedule_regen)
        style_row.addWidget(self.style_combo, stretch=1)
        gv.addLayout(style_row)
        self.chk_gpu = QCheckBox("GPU compute terrain")
        self.chk_gpu.setToolTip("Requires OpenGL 4.3+. Terrain noise and "
                                "thermal erosion run on the GPU.")
        self.chk_gpu.setChecked(True)
        self.chk_gpu.setEnabled(False)
        self.chk_gpu.toggled.connect(lambda _: self._schedule_regen())
        gv.addWidget(self.chk_gpu)
        surprise = QPushButton("\U0001f3b2 Surprise me")
        surprise.clicked.connect(self._surprise)
        gv.addWidget(surprise)
        pgrid = QGridLayout()
        pgrid.setSpacing(4)
        for i, name in enumerate(PRESETS):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self._apply_preset(PRESETS[n]))
            pgrid.addWidget(btn, i // 4, i % 4)
        gv.addLayout(pgrid)
        v.addWidget(g)

        g = QGroupBox("Terrain")
        tl = QVBoxLayout(g)
        self._add_slider(tl, "continent", "Continents", 4, 40, 15, 0.1, "{:.1f}")
        self._add_slider(tl, "sea", "Sea level", -100, 50, 3, 0.01, "{:+.2f}")
        self._add_slider(tl, "mountains", "Mountains", 0, 35, 14, 0.01, "{:.2f}")
        self._add_slider(tl, "warp", "Warp", 0, 100, 40, 0.01, "{:.2f}")
        self._add_slider(tl, "detail", "Detail", 0, 30, 10, 0.1, "{:.1f}")
        self._add_slider(tl, "impacts", "Impacts", 0, 100, 0, 1, "{:.0f}")
        self._add_slider(tl, "volcanoes", "Volcanoes", 0, 40, 0, 1, "{:.0f}")
        self._add_slider(tl, "erosion", "Erosion (hydraulic)", 0, 30, 18, 1, "{:.0f}")
        self._add_slider(tl, "thermal", "Erosion (thermal)", 0, 40, 8, 1, "{:.0f}")
        self._add_slider(tl, "rivers", "Rivers", 0, 100, 65, 0.01, "{:.2f}")
        self._add_slider(tl, "adaptive", "Refine (adaptive)", 0, 2, 1, 1, "{:.0f}")
        self._add_slider(tl, "age", "Geologic age", 0, 500, 0, 1,
                         "{:.0f} Myr")
        v.addWidget(g)

        g = QGroupBox("Climate")
        cl = QVBoxLayout(g)
        self._add_slider(cl, "rain", "Rainfall", -40, 40, 0, 0.01, "{:+.2f}")
        self._add_slider(cl, "climate", "Climate", -30, 30, 0, 0.01, "{:+.2f}")
        self._add_slider(cl, "season", "Season", -100, 100, 0, 0.01, "{:+.2f}")
        self._add_slider(cl, "tf", "Terraform", 0, 100, 0, 0.01, "{:.2f}")
        v.addWidget(g)

        g = QGroupBox("Scene")
        sc = QVBoxLayout(g)
        self.chk_rotate = QCheckBox("Auto-rotate")
        self.chk_rotate.setChecked(True)
        self.chk_rotate.toggled.connect(self._set_rotate)
        self.chk_clouds = QCheckBox("Clouds")
        self.chk_clouds.setChecked(True)
        self.chk_clouds.toggled.connect(self._set_clouds)
        self.chk_atmo = QCheckBox("Atmosphere")
        self.chk_atmo.setChecked(True)
        self.chk_atmo.toggled.connect(self._set_atmo)
        self.chk_lights = QCheckBox("City lights / lava glow")
        self.chk_lights.setChecked(True)
        self.chk_lights.toggled.connect(self._set_lights)
        self.chk_moon = QCheckBox("Moons")
        self.chk_moon.setChecked(True)
        self.chk_moon.toggled.connect(self._set_moon)
        self.chk_veg = QCheckBox("Vegetation")
        self.chk_veg.setChecked(True)
        self.chk_veg.toggled.connect(self._set_veg)
        self.chk_wire = QCheckBox("Wireframe")
        self.chk_wire.toggled.connect(self._set_wire)
        self.chk_comp = QCheckBox("Binary companion")
        self.chk_comp.toggled.connect(self._set_companion)
        self.chk_mrings = QCheckBox("Moon rings")
        self.chk_mrings.toggled.connect(self._set_mrings)
        self.chk_matmo = QCheckBox("Moon atmospheres")
        self.chk_matmo.toggled.connect(self._set_matmo)
        self.chk_weather = QCheckBox("Weather particles")
        self.chk_weather.setChecked(True)
        self.chk_weather.toggled.connect(self._set_weather)
        for c in (self.chk_rotate, self.chk_clouds, self.chk_atmo,
                  self.chk_lights, self.chk_moon, self.chk_veg, self.chk_comp,
                  self.chk_mrings, self.chk_matmo, self.chk_weather,
                  self.chk_wire):
            sc.addWidget(c)
        self._add_slider(sc, "moons", "Moon count", 0, 3, 1, 1, "{:.0f}", live=True)
        self._add_slider(sc, "rings", "Rings", 0, 100, 0, 0.01, "{:.2f}", live=True)
        self._add_slider(sc, "cover", "Cloud cover", 0, 100, 55, 0.01,
                         "{:.2f}", live=True)
        self._add_slider(sc, "tilt", "Axial tilt", -45, 45, 23, 1,
                         "{:.0f}\u00b0", live=True)
        self._add_slider(sc, "speed", "Time speed", 0, 6, 2, 0.5,
                         "{:.1f}\u00d7", live=True)
        self._add_slider(sc, "veg", "Forest density", 0, 100, 70, 0.01,
                         "{:.2f}", live=True)
        self.chk_hires = QCheckBox("Auto hi-res on zoom")
        self.chk_hires.toggled.connect(self._set_hires)
        sc.addWidget(self.chk_hires)
        v.addWidget(g)

        g = QGroupBox("Effects")
        ef = QVBoxLayout(g)
        self.chk_aurora = QCheckBox("Aurora")
        self.chk_aurora.setChecked(True)
        self.chk_aurora.toggled.connect(self._set_aurora)
        self.chk_orbits = QCheckBox("Moon orbit paths")
        self.chk_orbits.setChecked(True)
        self.chk_orbits.toggled.connect(self._set_orbits)
        self.chk_cycle = QCheckBox("Day/night cycle")
        self.chk_cycle.toggled.connect(self._set_cycle)
        ef.addWidget(self.chk_aurora)
        ef.addWidget(self.chk_orbits)
        ef.addWidget(self.chk_cycle)
        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("Atmo quality"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(list(_QUALITY))
        self.quality_combo.setCurrentText("Medium")
        self.quality_combo.currentTextChanged.connect(self._set_quality)
        qrow.addWidget(self.quality_combo, stretch=1)
        ef.addLayout(qrow)
        self._add_slider(ef, "bloom", "Bloom", 0, 100, 35, 0.01, "{:.2f}", live=True)
        self._add_slider(ef, "god", "God rays", 0, 100, 35, 0.01, "{:.2f}",
                         live=True)
        self._add_slider(ef, "flare", "Lens flare", 0, 100, 60, 0.01,
                         "{:.2f}", live=True)
        self._add_slider(ef, "multi", "Multi-scatter", 0, 100, 30, 0.01,
                         "{:.2f}", live=True)
        self._add_slider(ef, "relief", "Relief detail", 0, 80, 28, 0.01,
                         "{:.2f}", live=True)
        self._add_slider(ef, "atmo", "Atmo density", 10, 250, 130, 0.01,
                         "{:.2f}", live=True)
        self._add_slider(ef, "cyclespd", "Cycle speed", 1, 20, 4, 1,
                         "{:.0f}", live=True)
        v.addWidget(g)

        g = QGroupBox("Sun")
        su = QVBoxLayout(g)
        self._add_slider(su, "sunaz", "Azimuth", 0, 360, 40, 1, "{:.0f}\u00b0",
                         live=True)
        self._add_slider(su, "sunel", "Altitude", -30, 80, 15, 1, "{:.0f}\u00b0",
                         live=True)
        v.addWidget(g)

        g = QGroupBox("Export")
        ex = QGridLayout(g)
        b1 = QPushButton("Screenshot")
        b1.clicked.connect(self._screenshot)
        b2 = QPushButton("OBJ")
        b2.clicked.connect(self._export_obj)
        b3 = QPushButton("glTF (glb)")
        b3.clicked.connect(self._export_glb)
        b4 = QPushButton("Map pack")
        b4.clicked.connect(self._export_map)
        b5 = QPushButton("Turntable GIF")
        b5.clicked.connect(self._export_turntable)
        b6 = QPushButton("Skybox (6 PNG)")
        b6.clicked.connect(self._export_skybox)
        for i, b in enumerate((b1, b2, b3, b4, b5, b6)):
            ex.addWidget(b, i // 2, i % 2)
        v.addWidget(g)

        g = QGroupBox("Config")
        cf = QHBoxLayout(g)
        b1 = QPushButton("Save\u2026")
        b1.clicked.connect(self._config_save)
        b2 = QPushButton("Load\u2026")
        b2.clicked.connect(self._config_load)
        b3 = QPushButton("Copy")
        b3.clicked.connect(self._config_copy)
        cf.addWidget(b1)
        cf.addWidget(b2)
        cf.addWidget(b3)
        v.addWidget(g)

        v.addStretch(1)
        self.stats = QLabel("generating\u2026")
        self.stats.setStyleSheet("color: #9aa0aa;")
        self.stats.setWordWrap(True)
        v.addWidget(self.stats)
        hint = QLabel("drag: orbit \u00b7 wheel: zoom \u00b7 dbl-click: reset\n"
                      "right-click: survey \u00b7 P: photo \u00b7 F: fullscreen\n"
                      "drop a screenshot here to reproduce its planet")
        hint.setStyleSheet("color: #6b7078;")
        v.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedWidth(410)
        return scroll

    def _add_slider(self, layout, key, label, lo, hi, default, scale, fmt,
                    live=False):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        value = QLabel()
        value.setMinimumWidth(56)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def sync(val, lab=value, sc=scale, f=fmt):
            lab.setText(f.format(val * sc))

        slider.valueChanged.connect(sync)
        slider.valueChanged.connect(self._live_sync if live
                                    else (lambda _: self._schedule_regen()))
        sync(slider.value())
        row.addWidget(slider, stretch=1)
        row.addWidget(value)
        layout.addLayout(row)
        self._sliders[key] = (slider, scale)

    def _make_shortcuts(self):
        sc = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc.activated.connect(self.chk_rotate.toggle)
        sc = QShortcut(QKeySequence("R"), self)
        sc.activated.connect(
            lambda: self.seed_box.setValue(random.randint(1, 999999)))
        for key, chk in (("W", "chk_wire"), ("C", "chk_clouds"),
                         ("L", "chk_lights"), ("A", "chk_aurora"),
                         ("O", "chk_orbits")):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(getattr(self, chk).toggle)
        sc = QShortcut(QKeySequence("P"), self)
        sc.activated.connect(self._toggle_photo)
        sc = QShortcut(QKeySequence("F"), self)
        sc.activated.connect(self._toggle_fullscreen)
        sc = QShortcut(QKeySequence("Ctrl+S"), self)
        sc.activated.connect(self._screenshot)

    def _toggle_photo(self):
        self.panel.setVisible(not self.panel.isVisible())

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ---- helpers ----
    def _val(self, key):
        slider, scale = self._sliders[key]
        return slider.value() * scale

    def _schedule_regen(self):
        self._regen_timer.start()

    def _subdiv_changed(self, v):
        self._base_subdiv = v
        self._schedule_regen()

    def _on_hires(self, hi):
        self.subdiv_box.blockSignals(True)
        self.subdiv_box.setValue(7 if hi else self._base_subdiv)
        self.subdiv_box.blockSignals(False)
        self._schedule_regen()

    def _live_sync(self, *_):
        g = self.gl_widget
        g.cloud_cover = self._val("cover")
        g.tilt = self._val("tilt")
        g.sun_az = self._val("sunaz")
        g.sun_el = self._val("sunel")
        g.ring_opacity = self._val("rings")
        g.moon_count = int(round(self._val("moons")))
        g.speed = self._val("speed")
        g.bloom = self._val("bloom")
        g.god_strength = self._val("god")
        g.flare = self._val("flare")
        g.multi = self._val("multi")
        g.relief_strength = self._val("relief")
        g.atmo_density = self._val("atmo")
        g.sun_cycle_speed = self._val("cyclespd")
        g.veg_density = self._val("veg")
        g.terraform = self._val("tf")
        g.update()

    def _set_quality(self, name):
        self.gl_widget.set_atmo_quality(name)

    def _set_rotate(self, c): self.gl_widget.auto_rotate = c
    def _set_clouds(self, c): self.gl_widget.show_clouds = c; self.gl_widget.update()
    def _set_atmo(self, c): self.gl_widget.show_atmo = c; self.gl_widget.update()
    def _set_lights(self, c): self.gl_widget.show_lights = c; self.gl_widget.update()
    def _set_moon(self, c): self.gl_widget.show_moon = c; self.gl_widget.update()
    def _set_wire(self, c): self.gl_widget.wireframe = c; self.gl_widget.update()
    def _set_aurora(self, c): self.gl_widget.show_aurora = c; self.gl_widget.update()
    def _set_orbits(self, c): self.gl_widget.show_orbits = c; self.gl_widget.update()
    def _set_veg(self, c): self.gl_widget.show_veg = c; self.gl_widget.update()
    def _set_companion(self, c): self.gl_widget.show_companion = c; self.gl_widget.update()
    def _set_mrings(self, c): self.gl_widget.show_moon_rings = c; self.gl_widget.update()
    def _set_matmo(self, c): self.gl_widget.show_moon_atmo = c; self.gl_widget.update()
    def _set_weather(self, c): self.gl_widget.show_weather = c; self.gl_widget.update()
    def _set_cycle(self, c): self.gl_widget.sun_cycle = c
    def _set_hires(self, c): self.gl_widget.auto_hires = c

    def _sun_ui(self, az):
        s, _ = self._sliders["sunaz"]
        s.blockSignals(True)
        s.setValue(int(round(az)) % 360)
        s.blockSignals(False)

    def _on_fps(self, fps):
        self.setWindowTitle(f"Planet Generator v8 \u2014 {fps} fps")

    def _on_hover(self, lat, lon):
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        self.hover_lbl.setText(
            f"cursor: {abs(lat):.1f}\u00b0{ns} {abs(lon):.1f}\u00b0{ew}")

    def _on_probe(self, info):
        ns = "N" if info["lat"] >= 0 else "S"
        ew = "E" if info["lon"] >= 0 else "W"
        tc = info["temp"] * 45.0 - 12.0
        ek = info["elev"] * 6371.0
        elev_s = (f"depth {abs(ek):.1f} km" if ek < 0 else f"elev {ek:.1f} km")
        self.probe_lbl.setText(
            f"{info['biome']}\n"
            f"{abs(info['lat']):.1f}\u00b0{ns} {abs(info['lon']):.1f}\u00b0{ew}"
            f"  \u00b7  {elev_s}\n"
            f"temp {tc:.0f}\u00b0C  \u00b7  moisture {info['moist']*100:.0f}%")

    def _go_to(self):
        self.gl_widget.focus_latlon(self.lat_box.value(), self.lon_box.value())

    def _apply_preset(self, values):
        if "style" in values:
            i = self.style_combo.findText(values["style"].capitalize())
            if i >= 0:
                self.style_combo.setCurrentIndex(i)
        for key, val in values.items():
            if key in ("style", "clouds", "atmo") or key not in self._sliders:
                continue
            slider, scale = self._sliders[key]
            slider.setValue(int(round(val / scale)))
        if "clouds" in values:
            self.chk_clouds.setChecked(values["clouds"])
        if "atmo" in values:
            self.chk_atmo.setChecked(values["atmo"])
        self._schedule_regen()

    def _surprise(self):
        style = random.choices(
            ["terran", "terran", "ice", "mars", "moon", "lava", "gas"], k=1)[0]
        vals = dict(
            adaptive=random.randint(0, 2),
            continent=random.uniform(0.9, 3.4),
            sea=random.uniform(-0.35, 0.42),
            mountains=random.uniform(0.02, 0.28),
            warp=random.uniform(0.10, 0.75),
            detail=random.uniform(0.4, 1.5),
            impacts=int(random.uniform(0, 60)),
            volcanoes=int(random.uniform(0, 24)),
            rain=random.uniform(-0.32, 0.28),
            climate=random.uniform(-0.22, 0.22),
            erosion=int(random.uniform(0, 30)),
            thermal=int(random.uniform(0, 24)),
            rivers=random.uniform(0.0, 0.9),
            rings=(random.choice([0.0, 0.7, 0.85]) if style == "gas" else 0.0),
            moons=int(random.uniform(0, 3)),
        )
        i = self.style_combo.findText(style.capitalize())
        if i >= 0:
            self.style_combo.setCurrentIndex(i)
        for key, val in vals.items():
            slider, scale = self._sliders[key]
            slider.setValue(int(round(val / scale)))
        self.seed_box.setValue(random.randint(1, 999999))

    def _collect_params(self):
        return PlanetParams(
            seed=self.seed_box.value(),
            subdiv=self.subdiv_box.value(),
            adaptive=int(round(self._val("adaptive"))),
            continent_scale=self._val("continent"),
            sea_level=self._val("sea"),
            mountain_height=self._val("mountains"),
            warp=self._val("warp"),
            detail=self._val("detail"),
            impacts=int(round(self._val("impacts"))),
            volcanoes=int(round(self._val("volcanoes"))),
            erosion=self._val("erosion"),
            thermal=int(round(self._val("thermal"))),
            rivers=self._val("rivers"),
            rainfall=self._val("rain"),
            climate=self._val("climate"),
            season=self._val("season"),
            age=self._val("age"),
            terraform=self._val("tf"),
            style=self.style_combo.currentText().lower(),
        )

    def _apply_cli(self, args):
        if args.seed:
            self.seed_box.setValue(args.seed)
        if args.style:
            i = self.style_combo.findText(args.style.capitalize())
            if i >= 0:
                self.style_combo.setCurrentIndex(i)
        if args.subdiv:
            self.subdiv_box.setValue(args.subdiv)
        if args.preset and args.preset in PRESETS:
            self._apply_preset(PRESETS[args.preset])

    # ---- config ----
    def _config_dict(self):
        return dict(
            version=8,
            params=asdict(self._collect_params()),
            scene={k: slider.value() for k, (slider, _) in self._sliders.items()},
            checks=dict(rotate=self.chk_rotate.isChecked(),
                        clouds=self.chk_clouds.isChecked(),
                        atmo=self.chk_atmo.isChecked(),
                        lights=self.chk_lights.isChecked(),
                        moon=self.chk_moon.isChecked(),
                        veg=self.chk_veg.isChecked(),
                        companion=self.chk_comp.isChecked(),
                        mrings=self.chk_mrings.isChecked(),
                        matmo=self.chk_matmo.isChecked(),
                        weather=self.chk_weather.isChecked(),
                        aurora=self.chk_aurora.isChecked(),
                        orbits=self.chk_orbits.isChecked(),
                        cycle=self.chk_cycle.isChecked(),
                        hires=self.chk_hires.isChecked(),
                        quality=self.quality_combo.currentText(),
                        style=self.style_combo.currentText()),
        )

    def _config_apply(self, cfg):
        p = cfg.get("params", {})
        if "style" in p:
            i = self.style_combo.findText(p["style"].capitalize())
            if i >= 0:
                self.style_combo.setCurrentIndex(i)
        keymap = dict(seed="seed", subdiv="subdiv", adaptive="adaptive",
                      continent_scale="continent", sea_level="sea",
                      mountain_height="mountains", warp="warp", detail="detail",
                      impacts="impacts", volcanoes="volcanoes", erosion="erosion",
                      thermal="thermal", rivers="rivers", rainfall="rain",
                      climate="climate", season="season", age="age",
                      terraform="tf")
        for pk, sk in keymap.items():
            if pk in p and sk in self._sliders:
                if sk == "seed":
                    self.seed_box.setValue(int(p[pk]))
                elif sk == "subdiv":
                    self.subdiv_box.setValue(int(p[pk]))
                else:
                    slider, scale = self._sliders[sk]
                    slider.setValue(int(round(p[pk] / scale)))
        for k, val in cfg.get("scene", {}).items():
            if k in self._sliders:
                self._sliders[k][0].setValue(int(val))
        checks = cfg.get("checks", {})
        cmap = dict(rotate=self.chk_rotate, clouds=self.chk_clouds,
                    atmo=self.chk_atmo, lights=self.chk_lights,
                    moon=self.chk_moon, veg=self.chk_veg,
                    companion=self.chk_comp, mrings=self.chk_mrings,
                    matmo=self.chk_matmo, weather=self.chk_weather,
                    aurora=self.chk_aurora, orbits=self.chk_orbits,
                    cycle=self.chk_cycle, hires=self.chk_hires)
        for k, chk in cmap.items():
            if k in checks:
                chk.setChecked(bool(checks[k]))
        if "quality" in checks:
            i = self.quality_combo.findText(checks["quality"])
            if i >= 0:
                self.quality_combo.setCurrentIndex(i)
        if "style" in checks:
            i = self.style_combo.findText(checks["style"])
            if i >= 0:
                self.style_combo.setCurrentIndex(i)
        self._schedule_regen()

    def _config_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "planet.json",
                                              "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._config_dict(), fh, indent=1)
        self.stats.setText("config saved")

    def _config_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self._config_apply(json.load(fh))
        except Exception as e:
            QMessageBox.critical(self, "Config error", str(e))

    def _config_copy(self):
        QApplication.clipboard().setText(json.dumps(self._config_dict()))
        self.stats.setText("config copied to clipboard")

    # ---- generation (progressive + optional GPU) ----
    def _regen(self):
        if self._worker is not None:
            self._pending = True
            return
        params = self._collect_params()
        # progressive: instant low-res preview while the full build runs
        if params.subdiv > 4:
            pv = PlanetParams(**{**asdict(params), "subdiv": 4, "adaptive": 0})
            t0 = time.perf_counter()
            pdata = generate_planet(pv)
            pdata["_ms"] = (time.perf_counter() - t0) * 1000.0
            pdata["preview"] = True
            self.gl_widget.set_planet(pdata)

        gpu_ok = (self.chk_gpu.isEnabled() and self.chk_gpu.isChecked()
                  and self.gl_widget.has_compute and params.style != "gas"
                  and params.adaptive == 0)
        gpu_elev = gpu_glow = None
        gpu_tag = ""
        if gpu_ok:
            verts, faces = get_icosphere(params.subdiv)
            res = self.gl_widget.gpu_terrain(params, verts)
            if res is not None:
                gpu_elev, gpu_glow = res
                gpu_tag = " \u00b7 GPU"
        self.stats.setText("generating\u2026")
        self._worker = GenWorker(params, gpu_elev, gpu_glow)
        self._worker.finished_gen.connect(self._on_generated)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()
        self._gpu_tag = gpu_tag

    def _on_generated(self, data):
        w, self._worker = self._worker, None
        w.deleteLater()
        self.gl_widget.set_planet(data)
        st = data.get("stats", {})
        ocean = st.get("ocean", 0.0)
        self.name_lbl.setText(
            (data.get("name", "\u2014"))
            + (" (preview)" if data.get("preview") else ""))
        self.meta_lbl.setText(
            f"{data['style']} \u00b7 seed {data['seed']} \u00b7 "
            f"{data['n_vertices']:,} verts{self._gpu_tag}")
        self.stat_lbl.setText(
            f"Ocean {ocean*100:.0f}%  \u00b7  Land {(1-ocean)*100:.0f}%  \u00b7  "
            f"Lakes {st.get('lakes', 0)*100:.1f}%\n"
            f"Peak {st.get('peak', 0.0)*6371:.0f} km  \u00b7  "
            f"Glaciers {st.get('glaciers', 0)*100:.1f}%  \u00b7  "
            f"Trees {st.get('trees', 0):,}")
        mm = data.get("_minimap")
        if mm is not None:
            self.map_lbl.setPixmap(
                QPixmap.fromImage(mm).scaledToWidth(354,
                                                     Qt.SmoothTransformation))
        self.stats.setText(f"generated in {data['_ms']:.0f} ms "
                           f"\u00b7 {data['n_faces']:,} triangles"
                           + ("" if not data.get("preview")
                              else " \u00b7 building full resolution\u2026"))
        if self._pending:
            self._pending = False
            self._regen()

    def _on_failed(self, msg):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self.stats.setText("generation failed")
        QMessageBox.critical(self, "Generation error", msg)

    # ---- exports ----
    def _screenshot(self):
        img = self.gl_widget.grabFramebuffer()
        name = time.strftime("planet_%Y%m%d_%H%M%S.png")
        if img.save(name, "PNG"):
            insert_png_text(name, "planetConfig",
                            json.dumps(self._config_dict()))
            self.stats.setText(f"saved {name} (config embedded \u2014 "
                               f"drop it back to reproduce)")

    def _export_obj(self):
        if self.gl_widget.planet is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "planet.obj",
                                              "Wavefront OBJ (*.obj)")
        if path:
            self._start_export(ExportWorker("obj", path, data=self.gl_widget.planet))

    def _export_glb(self):
        if self.gl_widget.planet is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export glTF binary",
                                              "planet.glb", "glTF binary (*.glb)")
        if path:
            self._start_export(ExportWorker("glb", path, data=self.gl_widget.planet))

    def _export_map(self):
        if self.gl_widget.planet is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export map pack",
                                              "planet_map.png", "PNG (*.png)")
        if path:
            self._start_export(ExportWorker("map", path, data=self.gl_widget.planet))

    def _export_turntable(self):
        if self.gl_widget.planet is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export turntable",
                                              "planet_turntable.gif",
                                              "GIF (*.gif)")
        if not path:
            return
        w = self.gl_widget
        base_yaw = w.yaw
        frames = []
        try:
            for i in range(48):
                w.yaw = (base_yaw + i * 7.5) % 360.0
                w.update()
                QApplication.processEvents()
                img = w.grabFramebuffer().scaled(
                    512, 288, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                ).convertToFormat(QImage.Format_RGBA8888)
                h, wd = img.height(), img.width()
                arr = np.frombuffer(img.constBits(), np.uint8).reshape(
                    h, img.bytesPerLine())[:, :wd * 4].reshape(h, wd, 4)[:, :, :3]
                frames.append(np.ascontiguousarray(arr))
        finally:
            w.yaw = base_yaw
            w.update()
        try:
            import imageio
            imageio.mimsave(path, frames, fps=16, loop=0)
            self.stats.setText(f"GIF saved: {path} (48 frames)")
        except ImportError:
            folder = path + "_frames"
            os.makedirs(folder, exist_ok=True)
            for i, fr in enumerate(frames):
                QImage(fr.tobytes(), fr.shape[1], fr.shape[0], 3 * fr.shape[1],
                       QImage.Format_RGB888).save(
                    os.path.join(folder, f"f{i:03d}.png"), "PNG")
            self.stats.setText(f"imageio missing \u2014 saved PNGs to {folder}")

    def _export_skybox(self):
        folder = QFileDialog.getExistingDirectory(self, "Skybox output folder")
        if not folder:
            return
        paths = self.gl_widget.export_skybox(folder)
        self.stats.setText(f"skybox: {len(paths)} faces saved to {folder}")

    def _start_export(self, worker):
        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.information(self, "Busy", "An export is already running.")
            worker.deleteLater()
            return
        self._export_worker = worker
        worker.finished_export.connect(self._on_export_done)
        worker.failed.connect(
            lambda m: QMessageBox.critical(self, "Export error", m))
        worker.start()

    def _on_export_done(self, kind, msg):
        if self._export_worker is not None:
            self._export_worker.deleteLater()
            self._export_worker = None
        QMessageBox.information(self, "Export complete", msg)

    def closeEvent(self, e):
        if not self._used_cli:
            try:
                with open(_CFG_PATH, "w", encoding="utf-8") as fh:
                    json.dump(self._config_dict(), fh)
            except Exception:
                pass
        for w in (self._worker, self._export_worker):
            if w is not None and w.isRunning():
                w.wait(3000)
        super().closeEvent(e)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def apply_dark_palette(app):
    p = app.palette()
    p.setColor(QPalette.Window, QColor(28, 30, 36))
    p.setColor(QPalette.WindowText, QColor(220, 222, 228))
    p.setColor(QPalette.Base, QColor(20, 22, 27))
    p.setColor(QPalette.Text, QColor(220, 222, 228))
    p.setColor(QPalette.Button, QColor(37, 40, 47))
    p.setColor(QPalette.ButtonText, QColor(220, 222, 228))
    p.setColor(QPalette.Highlight, QColor(52, 110, 190))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipBase, QColor(40, 43, 51))
    p.setColor(QPalette.ToolTipText, QColor(220, 222, 228))
    p.setColor(QPalette.PlaceholderText, QColor(120, 124, 132))
    app.setPalette(p)


def main():
    ap = argparse.ArgumentParser(description="Procedural planet generator")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--style", choices=["terran", "ice", "mars", "lava",
                                        "moon", "gas"])
    ap.add_argument("--subdiv", type=int, choices=range(2, 10))
    ap.add_argument("--preset", choices=list(PRESETS))
    args = ap.parse_args()

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_dark_palette(app)

    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()