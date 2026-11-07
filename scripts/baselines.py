"""baselines.py — líneas base del agente (Persona 4, P4.5).

Antes de entrenar conviene saber qué progreso consiguen dos agentes que no
aprenden, para tener con qué comparar al PPO:

- Aleatorio: elige una de las 7 acciones al azar en cada paso.
- Heurístico: el mismo bot que valida los niveles (mario_game.validate.Bot),
  que corre a la derecha y salta ante paredes, fosos y goombas.

Las dos líneas base usan el entorno real (MarioPygame-v0) a través del contrato,
así que el "progreso medio" se mide igual que para el PPO: con la clave
`progress` del `info` (INTERFACES.md §7).

    python scripts/baselines.py --episodes 50
    python scripts/baselines.py --episodes 50 --level 2

Imprime una tabla con el progreso medio y la tasa de bandera de cada línea base.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Permite ejecutar el script directamente (python scripts/baselines.py) con los
# módulos de la raíz del repo (mario_env, mario_game) en el path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import mario_env  # noqa: E402  registra el entorno
from mario_env import ACTIONS, ENV_ID  # noqa: E402
from mario_game.validate import Bot  # noqa: E402

# El bot heurístico devuelve (left, right, jump, run); hay que traducirlo al
# índice de acción del entorno (INTERFACES.md §7).
_ACTION_INDEX = {combo: i for i, combo in enumerate(ACTIONS)}


def _bot_action_index(game) -> int:
    """La acción del bot heurístico como índice de MarioEnv. Si la combinación
    del bot no está entre las 7 acciones, usa la más parecida a la derecha."""
    combo = Bot().act(game)
    if combo in _ACTION_INDEX:
        return _ACTION_INDEX[combo]
    left, right, jump, run = combo
    if right and jump:
        return 4 if run else 2
    if right:
        return 3 if run else 1
    return 5 if jump else 0


def run_random(env, episodes, rng):
    progress, flags = [], 0
    for ep in range(episodes):
        _obs, _info = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        info = {}
        while not done:
            _obs, _r, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
            done = term or trunc
        progress.append(float(info.get("progress", 0.0)))
        flags += bool(info.get("flag_get"))
    return progress, flags


def run_heuristic(env, episodes, rng):
    progress, flags = [], 0
    for ep in range(episodes):
        _obs, _info = env.reset(seed=int(rng.integers(1_000_000)))
        # El bot mantiene su propio estado de salto entre frames.
        bot = Bot()
        done = False
        info = {}
        while not done:
            combo = bot.act(env.unwrapped.game)
            idx = _ACTION_INDEX.get(combo)
            if idx is None:
                left, right, jump, run = combo
                idx = (4 if run else 2) if right and jump else \
                      (3 if run else 1) if right else (5 if jump else 0)
            _obs, _r, term, trunc, info = env.step(idx)
            done = term or trunc
        progress.append(float(info.get("progress", 0.0)))
        flags += bool(info.get("flag_get"))
    return progress, flags


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Líneas base aleatoria y heurística para Mario (P4.5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50, help="Episodios por línea base.")
    parser.add_argument("--level", type=int, default=1, help="Nivel (1 o 2).")
    parser.add_argument("--seed", type=int, default=0, help="Semilla.")
    args = parser.parse_args(argv)

    import gymnasium as gym

    env = gym.make(ENV_ID, obs_type="grid", level=args.level)
    rng = np.random.default_rng(args.seed)

    rand_progress, rand_flags = run_random(env, args.episodes, rng)
    heur_progress, heur_flags = run_heuristic(env, args.episodes, rng)
    env.close()

    n = args.episodes
    print(f"\nLíneas base en el nivel {args.level} ({n} episodios cada una)\n")
    print(f"{'Línea base':<14}  {'progreso medio':>15}  {'tasa bandera':>13}")
    print("-" * 46)
    print(f"{'aleatorio':<14}  {np.mean(rand_progress) * 100:>14.1f}%  "
          f"{rand_flags}/{n} ({rand_flags / n * 100:.0f}%)")
    print(f"{'heurístico':<14}  {np.mean(heur_progress) * 100:>14.1f}%  "
          f"{heur_flags}/{n} ({heur_flags / n * 100:.0f}%)")


if __name__ == "__main__":
    main()
