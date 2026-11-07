"""train.py — entrenamiento con PPO para Mario DRL (Persona 4).

Entrena un agente PPO (Stable-Baselines3) sobre el entorno MarioPygame-v0, con
varios entornos en paralelo, checkpoints periódicos y registro en TensorBoard.
También sirve para probar el pipeline en CartPole (P4.1) antes de gastar horas
en Mario.

Contrato que usa (INTERFACES.md §7): el entorno `MarioPygame-v0` con las
observaciones `grid` (498 valores) y `pixels` (84×84×1), 7 acciones y el
diccionario `info` con `progress`, `flag_get`, etc. El callback de métricas
(P4.2) lee esas claves de `info`.

Ejemplos
--------
Probar el pipeline en CartPole (debe llegar a ~500 de recompensa, P4.1):

    python train.py --cartpole --steps 100000

Entrenamiento principal con la observación grid, 2 M de pasos (P4.6):

    python train.py --obs grid --steps 2000000 --name grid_s0 --seed 0

Con pixels y apilado de 4 frames (P4.7):

    python train.py --obs pixels --frame-stack 4 --steps 2000000 --name pixels_s0

Reanudar desde un checkpoint:

    python train.py --obs grid --steps 1000000 --name grid_s0 --resume models/grid_s0/final.zip

Todo lo que genera (modelos, TensorBoard, vídeos) va a carpetas ignoradas por
git (`models/`, `runs/`): no se versiona.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# pygame y SDL sin pantalla: entrenar no necesita ventana, y así funciona en
# servidores y en el CI. Hay que ponerlo antes de que nada importe pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Directorios de salida (ignorados por .gitignore).
MODELS_DIR = Path("models")
RUNS_DIR = Path("runs")


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrenamiento PPO para Mario DRL (Persona 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cartpole", action="store_true",
                   help="Probar el pipeline en CartPole-v1 en vez de Mario (P4.1).")
    p.add_argument("--obs", choices=["grid", "pixels"], default="grid",
                   help="Tipo de observación de Mario.")
    p.add_argument("--level", type=int, default=1,
                   help="Nivel de Mario (1 o 2).")
    p.add_argument("--steps", type=int, default=2_000_000,
                   help="Pasos totales de entrenamiento.")
    p.add_argument("--n-envs", type=int, default=8,
                   help="Entornos en paralelo (P4.4: ajustar según la máquina).")
    p.add_argument("--seed", type=int, default=0, help="Semilla (P4.7).")
    p.add_argument("--name", default=None,
                   help="Nombre del experimento (carpeta de modelos y logs). "
                        "Por defecto se deduce de los parámetros.")
    p.add_argument("--frame-stack", type=int, default=1,
                   help="Frames apilados (VecFrameStack). Útil con pixels (P4.7).")
    p.add_argument("--random-start", action="store_true",
                   help="Empezar a veces desde un checkpoint al azar (P3.7, P4.7).")
    p.add_argument("--random-start-prob", type=float, default=0.5,
                   help="Probabilidad de inicio aleatorio cuando --random-start.")
    p.add_argument("--sticky-prob", type=float, default=0.0,
                   help="Acciones pegajosas para robustez (P3.9).")
    p.add_argument("--goomba-speed-min", type=float, default=None,
                   help="Velocidad mínima de goombas (con --goomba-speed-max, P3.9).")
    p.add_argument("--goomba-speed-max", type=float, default=None,
                   help="Velocidad máxima de goombas (con --goomba-speed-min, P3.9).")
    p.add_argument("--vec", choices=["subproc", "dummy"], default="subproc",
                   help="Vectorización: subproc (procesos, rápido) o dummy (un proceso).")
    p.add_argument("--n-steps", type=int, default=512,
                   help="Pasos por entorno antes de cada actualización de PPO.")
    p.add_argument("--batch-size", type=int, default=256, help="Tamaño del minibatch.")
    p.add_argument("--learning-rate", type=float, default=2.5e-4,
                   help="Tasa de aprendizaje.")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="Coeficiente de entropía (fomenta explorar).")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="Hilos de PyTorch (P4.4: con pocos hilos suele ir más rápido). "
                        "Por defecto no se tocan.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Dispositivo de PyTorch (P4.4: CPU frente a GPU).")
    p.add_argument("--save-freq", type=int, default=100_000,
                   help="Guardar un checkpoint cada tantos pasos (por entorno se divide).")
    p.add_argument("--resume", default=None,
                   help="Ruta de un .zip para continuar un entrenamiento.")
    return p


def _experiment_name(args) -> str:
    if args.name:
        return args.name
    if args.cartpole:
        return f"cartpole_s{args.seed}"
    parts = [f"lvl{args.level}", args.obs, f"s{args.seed}"]
    if args.frame_stack > 1:
        parts.append(f"fs{args.frame_stack}")
    if args.random_start:
        parts.append("rnd")
    return "_".join(parts)


def main(argv=None) -> None:
    args = _make_parser().parse_args(argv)

    # Imports pesados aquí dentro: así `python train.py --help` es instantáneo y
    # no obliga a tener torch instalado para leer la ayuda.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from mario_training import (MarioMetricsCallback, make_cartpole_venv,
                                make_mario_venv)

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)   # P4.4

    name = _experiment_name(args)
    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = RUNS_DIR
    tb_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    # Un checkpoint cada save_freq pasos de entorno (SB3 lo cuenta por entorno).
    save_freq = max(args.save_freq // args.n_envs, 1)
    callbacks.append(CheckpointCallback(
        save_freq=save_freq, save_path=str(model_dir), name_prefix="ckpt"))

    if args.cartpole:
        # Prueba del pipeline (P4.1): CartPole-v1, que llega a 500 de recompensa.
        venv = make_cartpole_venv(n_envs=args.n_envs, seed=args.seed, vec=args.vec)
        policy = "MlpPolicy"
    else:
        goomba_range = None
        if args.goomba_speed_min is not None and args.goomba_speed_max is not None:
            goomba_range = (args.goomba_speed_min, args.goomba_speed_max)
        venv = make_mario_venv(
            n_envs=args.n_envs, seed=args.seed, obs_type=args.obs, level=args.level,
            frame_stack=args.frame_stack, random_start=args.random_start,
            random_start_prob=args.random_start_prob, sticky_prob=args.sticky_prob,
            goomba_speed_range=goomba_range, vec=args.vec,
        )
        # CnnPolicy para imágenes (pixels), MlpPolicy para el vector grid.
        policy = "CnnPolicy" if args.obs == "pixels" else "MlpPolicy"
        # Las métricas Mario (P4.2) solo tienen sentido con el entorno Mario.
        callbacks.append(MarioMetricsCallback())

    if args.resume:
        print(f"Reanudando desde {args.resume}")
        model = PPO.load(args.resume, env=venv, device=args.device,
                         tensorboard_log=str(tb_dir))
    else:
        model = PPO(
            policy, venv, seed=args.seed, device=args.device,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, ent_coef=args.ent_coef,
            tensorboard_log=str(tb_dir), verbose=1,
        )

    print(f"Experimento: {name}  |  entornos: {args.n_envs} ({args.vec})  |  "
          f"dispositivo: {model.device}  |  pasos: {args.steps:,}")
    print(f"Modelos en {model_dir}/   TensorBoard en {tb_dir}/ (tensorboard --logdir {tb_dir})")

    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    tb_log_name=name, reset_num_timesteps=not args.resume,
                    progress_bar=True)
    finally:
        final_path = model_dir / "final.zip"
        model.save(str(final_path))
        venv.close()
        print(f"Modelo final guardado en {final_path}")


if __name__ == "__main__":
    main()
