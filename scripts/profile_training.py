"""profile_training.py — perfilado del entrenamiento (Persona 4, P4.4).

Mide los pasos por segundo que alcanza PPO con distintas configuraciones, para
elegir la más rápida antes del entrenamiento largo (P4.6). El plan avisa de que
en el prototipo con 24 hilos de PyTorch se quedaba en ~800 pasos/s, y con 4
hilos subía a ~2500-3000: por eso aquí se barre el número de hilos y de entornos.

Cada configuración entrena unos pocos miles de pasos (sin que importe el
resultado) y mide el tiempo real.

    python scripts/profile_training.py --obs grid --steps 20000
    python scripts/profile_training.py --obs grid --threads 1 2 4 8 --n-envs 4 8 16

Imprime una tabla de pasos/s y marca la configuración más rápida.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _time_config(obs, n_envs, threads, steps, device, vec) -> float:
    """Pasos por segundo de una configuración. Un proceso de PPO corto."""
    import torch
    from stable_baselines3 import PPO

    from mario_training import make_mario_venv

    torch.set_num_threads(threads)
    venv = make_mario_venv(n_envs=n_envs, seed=0, obs_type=obs, vec=vec)
    policy = "CnnPolicy" if obs == "pixels" else "MlpPolicy"
    model = PPO(policy, venv, device=device, n_steps=256, batch_size=256, verbose=0)
    # Un learn de calentamiento corto no hace falta: medimos el learn principal,
    # que ya incluye la recogida de experiencia y la actualización.
    start = time.perf_counter()
    model.learn(total_timesteps=steps, progress_bar=False)
    elapsed = time.perf_counter() - start
    venv.close()
    return steps / elapsed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Perfilado de pasos/s de PPO en Mario (P4.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obs", choices=["grid", "pixels"], default="grid")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Pasos por configuración (cuantos más, más fiable y más lento).")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Valores de hilos de PyTorch a probar.")
    parser.add_argument("--n-envs", type=int, nargs="+", default=[4, 8],
                        help="Valores de número de entornos a probar.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--vec", choices=["subproc", "dummy"], default="subproc")
    args = parser.parse_args(argv)

    print(f"Perfilado: obs={args.obs}, device={args.device}, vec={args.vec}, "
          f"{args.steps:,} pasos por configuración\n")
    print(f"{'entornos':>8}  {'hilos':>6}  {'pasos/s':>10}")
    print("-" * 30)

    best = None
    for n_envs in args.n_envs:
        for threads in args.threads:
            sps = _time_config(args.obs, n_envs, threads, args.steps, args.device, args.vec)
            marca = ""
            if best is None or sps > best[0]:
                best = (sps, n_envs, threads)
                marca = "  <-- mejor hasta ahora"
            print(f"{n_envs:>8}  {threads:>6}  {sps:>10,.0f}{marca}")

    if best:
        print(f"\nMás rápida: {best[1]} entornos, {best[2]} hilos -> {best[0]:,.0f} pasos/s")
        print("Usa esos valores en train.py: --n-envs {} --torch-threads {}".format(
            best[1], best[2]))


if __name__ == "__main__":
    main()
