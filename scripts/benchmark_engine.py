"""Mide cuántos frames por segundo simula el motor, sin gráficos.

Objetivo derivado de P3.8: el entorno debe superar 5000 pasos/s y cada paso
son 4 frames, así que el motor necesita unos 20 000 frames/s.

Uso:  python scripts/benchmark_engine.py
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import make_level  # noqa: E402
from mario_game.game import MarioGame  # noqa: E402

# Nivel de prueba de 200 columnas con fosos, tuberías, bloques y 10 goombas.
ROWS_ASCII = [
    ("....??B?......" * 15)[:200],
    "." * 200,
    "." * 200,
    ("......P......." * 15)[:200],
    ("......P...HH.." * 15)[:200],
    ("##########...#" * 15)[:200],
    ("##########...#" * 15)[:200],
]
GOOMBA_COLS = [c for c in range(15, 200, 19) if c % 14 not in (10, 11, 12)]  # no en fosos


def bench(frames: int = 200_000, seed: int = 0) -> float:
    rng = random.Random(seed)
    game = MarioGame(make_level(ROWS_ASCII, goomba_cols=GOOMBA_COLS,
                                start_col=1, flag_col=195))
    # Acciones como las del agente: una de 7, mantenida 4 frames.
    actions = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1),
               (0, 1, 1, 1), (0, 0, 1, 0), (1, 0, 0, 0)]
    start = time.perf_counter()
    done = 0
    while done < frames:
        a = actions[rng.randrange(7)]
        for _ in range(4):
            game.step(*a)
        done += 4
        if game.dead or game.won:
            game.reset()
    return frames / (time.perf_counter() - start)


if __name__ == "__main__":
    fps = bench()
    print(f"{fps:,.0f} frames/s  ->  {fps / 4:,.0f} pasos de entorno/s (4 frames/paso)")
    print("objetivo: >= 20,000 frames/s (5,000 pasos/s, P3.8)")
