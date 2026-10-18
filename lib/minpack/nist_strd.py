# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""NIST's Statistical Reference Datasets for nonlinear regression: the 27
problems in lib/minpack/data (fetched unmodified from
https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/), read from their
own files, and each model as a Python function in SciPy's form f(x, b1, ...,
bk), written from the "Model:" line of its file. py2viv translates them for
vp.optimize; SciPy calls them as they are.

    load("Misra1a") -> Problem(name, x, y, starts, certified, cert_sd, rss, difficulty)

Nelson has two predictors (x1, x2) and a model for log(y); its `x` is an
array of shape (2, m) and its residual function is `nelson_res` (for
least_squares), not a model of one x.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

# in NIST's order: lower, average, higher difficulty
NAMES = ["Misra1a", "Chwirut2", "Chwirut1", "Lanczos3", "Gauss1", "Gauss2", "DanWood", "Misra1b",
         "Kirby2", "Hahn1", "Nelson", "MGH17", "Lanczos1", "Lanczos2", "Gauss3", "Misra1c", "Misra1d", "Roszman1",
         "ENSO", "MGH09", "Thurber", "BoxBOD", "Rat42", "MGH10", "Eckerle4", "Rat43", "Bennett5"]


@dataclass
class Problem:
    name: str
    x: np.ndarray            # (m,), or (2, m) for Nelson
    y: np.ndarray
    starts: list             # [start 1, start 2], arrays of k
    certified: np.ndarray
    cert_sd: np.ndarray
    rss: float
    difficulty: str


def _lines(spec, text):
    m = re.search(spec + r"\s*\(lines\s+(\d+)\s+to\s+(\d+)\)", text)
    return int(m.group(1)), int(m.group(2))


def load(name) -> Problem:
    text = (DATA / (name + ".dat")).read_text()
    lines = text.splitlines()
    a, b = _lines("Data", text)
    rows = np.array([[float(v) for v in ln.split()] for ln in lines[a - 1:b] if ln.split()])
    y = rows[:, 0]
    x = rows[:, 1] if rows.shape[1] == 2 else rows[:, 1:].T.copy()
    s1, s2, cert, sd = [], [], [], []
    for ln in lines:
        m = re.match(r"\s*b\d+\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", ln)
        if m:
            s1.append(float(m.group(1)))
            s2.append(float(m.group(2)))
            cert.append(float(m.group(3)))
            sd.append(float(m.group(4)))
    rss = float(re.search(r"Residual Sum of Squares:\s+(\S+)", text).group(1))
    diff = re.search(r"(Lower|Average|Higher) Level of Difficulty", text).group(1).lower()
    return Problem(name, x, y, [np.array(s1), np.array(s2)], np.array(cert), np.array(sd), rss, diff)


# The models, from each file's "Model:" line.

def misra1a(x, b1, b2):
    return b1 * (1.0 - np.exp(-b2 * x))


def chwirut(x, b1, b2, b3):
    return np.exp(-b1 * x) / (b2 + b3 * x)


def lanczos(x, b1, b2, b3, b4, b5, b6):
    return b1 * np.exp(-b2 * x) + b3 * np.exp(-b4 * x) + b5 * np.exp(-b6 * x)


def gauss(x, b1, b2, b3, b4, b5, b6, b7, b8):
    return b1 * np.exp(-b2 * x) + b3 * np.exp(-(x - b4) ** 2 / b5 ** 2) + b6 * np.exp(-(x - b7) ** 2 / b8 ** 2)


def danwood(x, b1, b2):
    return b1 * x ** b2


def misra1b(x, b1, b2):
    return b1 * (1.0 - (1.0 + b2 * x / 2.0) ** (-2.0))


def kirby2(x, b1, b2, b3, b4, b5):
    return (b1 + b2 * x + b3 * x ** 2) / (1.0 + b4 * x + b5 * x ** 2)


def rational3(x, b1, b2, b3, b4, b5, b6, b7):
    return (b1 + b2 * x + b3 * x ** 2 + b4 * x ** 3) / (1.0 + b5 * x + b6 * x ** 2 + b7 * x ** 3)


def mgh17(x, b1, b2, b3, b4, b5):
    return b1 + b2 * np.exp(-x * b4) + b3 * np.exp(-x * b5)


def misra1c(x, b1, b2):
    return b1 * (1.0 - (1.0 + 2.0 * b2 * x) ** (-0.5))


def misra1d(x, b1, b2):
    return b1 * b2 * x * ((1.0 + b2 * x) ** (-1.0))


def roszman1(x, b1, b2, b3, b4):
    return b1 - b2 * x - np.arctan(b3 / (x - b4)) / np.pi


def enso(x, b1, b2, b3, b4, b5, b6, b7, b8, b9):
    return (b1 + b2 * np.cos(2.0 * np.pi * x / 12.0) + b3 * np.sin(2.0 * np.pi * x / 12.0)
            + b5 * np.cos(2.0 * np.pi * x / b4) + b6 * np.sin(2.0 * np.pi * x / b4)
            + b8 * np.cos(2.0 * np.pi * x / b7) + b9 * np.sin(2.0 * np.pi * x / b7))


def mgh09(x, b1, b2, b3, b4):
    return b1 * (x ** 2 + x * b2) / (x ** 2 + x * b3 + b4)


def boxbod(x, b1, b2):
    return b1 * (1.0 - np.exp(-b2 * x))


def rat42(x, b1, b2, b3):
    return b1 / (1.0 + np.exp(b2 - b3 * x))


def mgh10(x, b1, b2, b3):
    return b1 * np.exp(b2 / (x + b3))


def eckerle4(x, b1, b2, b3):
    return (b1 / b2) * np.exp(-0.5 * ((x - b3) / b2) ** 2)


def rat43(x, b1, b2, b3, b4):
    return b1 / ((1.0 + np.exp(b2 - b3 * x)) ** (1.0 / b4))


def bennett5(x, b1, b2, b3):
    return b1 * (b2 + x) ** (-1.0 / b3)


def nelson_res(b, x1, x2, y):
    """Nelson's residuals, log(y) - (b1 - b2 x1 exp(-b3 x2)), for least_squares."""
    return np.log(y) - (b[0] - b[1] * x1 * np.exp(-b[2] * x2))


MODELS = {
    "Misra1a": misra1a, "Chwirut2": chwirut, "Chwirut1": chwirut, "Lanczos3": lanczos, "Gauss1": gauss,
    "Gauss2": gauss, "DanWood": danwood, "Misra1b": misra1b, "Kirby2": kirby2, "Hahn1": rational3,
    "MGH17": mgh17, "Lanczos1": lanczos, "Lanczos2": lanczos, "Gauss3": gauss, "Misra1c": misra1c,
    "Misra1d": misra1d, "Roszman1": roszman1, "ENSO": enso, "MGH09": mgh09, "Thurber": rational3,
    "BoxBOD": boxbod, "Rat42": rat42, "MGH10": mgh10, "Eckerle4": eckerle4, "Rat43": rat43, "Bennett5": bennett5,
}


def lre(got, cert):
    """NIST's log relative error, the number of significant digits that agree:
    -log10(|got - cert| / |cert|), capped at 15 for an exact match, 0 if
    worse than one digit."""
    got, cert = np.atleast_1d(np.asarray(got, dtype=float)), np.atleast_1d(np.asarray(cert, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(got - cert) / np.abs(cert)
        out = np.where(r == 0, 15.0, -np.log10(r))
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, 0.0, 15.0)
