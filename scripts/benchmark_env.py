"""Mide cuántos pasos por segundo da el entorno con la observación grid.

Objetivo de P3.8: más de 5000 pasos/s con el motor real (cada paso son 4 frames).
Termina con código 1 si no llega, para que el CI lo detecte.

Uso:  python scripts/benchmark_env.py [--seconds 3] [--obs grid|pixels] [--level 1|2]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mario_env import MarioEnv  # noqa: E402

TARGET_STEPS_PER_S = 5000


def bench(seconds: float = 3.0, obs_type: str = "grid", level: int = 1,
          seed: int = 0) -> tuple[float, int]:
    """Pasos/s con acciones aleatorias (una de 7) y reset() al acabar cada episodio."""
    env = MarioEnv(obs_type=obs_type, level=level)     # el motor real
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    steps = episodes = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        _, _, terminated, truncated, _ = env.step(int(rng.integers(7)))
        steps += 1
        if terminated or truncated:
            env.reset()
            episodes += 1
    env.close()
    return steps / (time.perf_counter() - start), episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--obs", choices=("grid", "pixels"), default="grid")
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    if args.obs == "pixels":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    rate, episodes = bench(args.seconds, args.obs, args.level)
    print(f"{rate:,.0f} pasos de entorno/s con obs={args.obs}, nivel {args.level} "
          f"({episodes} episodios)")
    if args.obs == "grid":
        print(f"objetivo: > {TARGET_STEPS_PER_S:,} pasos/s (P3.8)")
        if rate <= TARGET_STEPS_PER_S:
            sys.exit(1)
