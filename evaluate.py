"""evaluate.py — evaluación de un agente de Mario DRL (Persona 4, P4.3).

Dos modos:

- Estadístico (por defecto): juega N episodios sin ventana e imprime la tasa de
  éxito (llegadas a la bandera), el progreso medio y el reparto de causas de
  muerte. Imprime también una línea por episodio (progreso, pasos, resultado).
- Visual (`--render`): abre una ventana y juega unos pocos episodios para verlo.

Carga un modelo PPO guardado por `train.py`. La observación (`grid`/`pixels`),
el apilado de frames y el nivel deben coincidir con los del entrenamiento; si no
coinciden, SB3 avisa del desajuste de formas.

Para medir la robustez (P4.8) se pasan `--sticky-prob` y/o el rango de velocidad
de goombas, o se evalúa en el nivel 2 con `--level 2`.

Ejemplos
--------
Estadístico, 20 episodios, el criterio de éxito del plan (P4.6/P4.8):

    python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20

Robustez: con aleatoriedad activada (P4.8):

    python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20 \
        --sticky-prob 0.1 --goomba-speed-min 0.3 --goomba-speed-max 1.5

Generalización: en el nivel 2 (P4.8):

    python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20 --level 2

Visual:

    python evaluate.py --model models/lvl1_grid_s0/final.zip --render --episodes 3
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # se cambia si --render
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluación de un agente PPO de Mario (Persona 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Ruta del .zip del modelo PPO.")
    p.add_argument("--obs", choices=["grid", "pixels"], default="grid",
                   help="Tipo de observación (debe coincidir con el entrenamiento).")
    p.add_argument("--level", type=int, default=1, help="Nivel (1 o 2, P4.8).")
    p.add_argument("--episodes", type=int, default=20,
                   help="Episodios a jugar (el plan evalúa con 20).")
    p.add_argument("--frame-stack", type=int, default=1,
                   help="Frames apilados (igual que en el entrenamiento).")
    p.add_argument("--render", action="store_true",
                   help="Abrir ventana y jugar en vez del modo estadístico.")
    p.add_argument("--deterministic", action="store_true",
                   help="Acciones deterministas (argmax) en vez de muestreadas.")
    p.add_argument("--seed", type=int, default=0, help="Semilla de evaluación.")
    p.add_argument("--start-col", type=int, default=None,
                   help="Empezar en una columna concreta (evaluar un tramo).")
    p.add_argument("--sticky-prob", type=float, default=0.0,
                   help="Acciones pegajosas para medir robustez (P4.8).")
    p.add_argument("--goomba-speed-min", type=float, default=None,
                   help="Velocidad mínima de goombas (con --goomba-speed-max).")
    p.add_argument("--goomba-speed-max", type=float, default=None,
                   help="Velocidad máxima de goombas (con --goomba-speed-min).")
    return p


def _goomba_range(args):
    if args.goomba_speed_min is not None and args.goomba_speed_max is not None:
        return (args.goomba_speed_min, args.goomba_speed_max)
    return None


def evaluate_stats(model, venv, episodes, deterministic, seed):
    """Juega `episodes` episodios en un VecEnv de un solo entorno y devuelve una
    lista de dicts por episodio con progreso, pasos, bandera y causa de muerte.

    Usa el mismo VecEnv que el entrenamiento (make_mario_venv), así la forma de
    la observación coincide siempre, también con pixels y apilado de frames. El
    VecEnv resetea solo al terminar un episodio (autoreset de SB3): recogemos un
    resultado cada vez que `dones[0]` es True."""
    venv.seed(seed)
    obs = venv.reset()
    results = []
    steps = 0
    total_reward = 0.0
    while len(results) < episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = venv.step(action)
        total_reward += float(reward[0])
        steps += 1
        if dones[0]:
            info = infos[0]
            results.append({
                "progress": float(info.get("progress", 0.0)),
                "steps": int(info.get("steps", steps)),
                "flag": bool(info.get("flag_get")),
                "death_reason": info.get("death_reason"),
                "reward": total_reward,
            })
            steps = 0
            total_reward = 0.0
    return results


def print_report(results) -> None:
    n = len(results)
    flags = sum(r["flag"] for r in results)
    mean_progress = sum(r["progress"] for r in results) / n
    mean_reward = sum(r["reward"] for r in results) / n
    deaths = Counter(r["death_reason"] for r in results if not r["flag"])

    print(f"\n{'ep':>3}  {'progreso':>9}  {'pasos':>6}  {'recompensa':>11}  resultado")
    print("-" * 52)
    for i, r in enumerate(results):
        outcome = "BANDERA" if r["flag"] else f"muerte:{r['death_reason']}"
        print(f"{i:>3}  {r['progress']*100:>8.1f}%  {r['steps']:>6}  "
              f"{r['reward']:>11.2f}  {outcome}")

    print("-" * 52)
    print(f"Episodios:        {n}")
    print(f"Tasa de exito:    {flags}/{n} = {flags / n * 100:.1f}% (llegadas a la bandera)")
    print(f"Progreso medio:   {mean_progress * 100:.1f}%")
    print(f"Recompensa media: {mean_reward:.2f}")
    if deaths:
        reparto = ", ".join(f"{k}: {v}" for k, v in deaths.most_common())
        print(f"Muertes:          {reparto}")
    # Criterio de exito del plan: >= 90% de 20 episodios.
    if flags / n >= 0.90:
        print("\n>> Cumple el criterio de exito del plan (>= 90%).")
    else:
        print("\n>> No llega al 90% del criterio del plan.")


def main(argv=None) -> None:
    args = _make_parser().parse_args(argv)

    if args.render:
        # Para ver la ventana hay que deshacer el driver dummy antes de pygame.
        os.environ.pop("SDL_VIDEODRIVER", None)
        os.environ.pop("SDL_AUDIODRIVER", None)

    from stable_baselines3 import PPO

    from mario_training import make_mario_venv

    render_mode = "human" if args.render else None
    goomba_range = _goomba_range(args)

    if args.render:
        # Modo visual: un solo entorno con ventana. No se puede usar SubprocVecEnv
        # (la ventana vive en el proceso principal), así que vec="dummy".
        venv = make_mario_venv(
            n_envs=1, seed=args.seed, obs_type=args.obs, level=args.level,
            frame_stack=args.frame_stack, sticky_prob=args.sticky_prob,
            goomba_speed_range=goomba_range, vec="dummy", render_mode=render_mode,
        )
        model = PPO.load(args.model, device="cpu")
        print(f"Modo visual: {args.episodes} episodios del modelo {args.model}")
        obs = venv.reset()
        episodes = 0
        flags = 0
        while episodes < args.episodes:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _reward, dones, infos = venv.step(action)
            if dones[0]:
                info = infos[0]
                won = bool(info.get("flag_get"))
                flags += won
                episodes += 1
                print(f"  episodio {episodes}: progreso {info.get('progress', 0)*100:.1f}%"
                      f"  {'BANDERA' if won else 'muerte:' + str(info.get('death_reason'))}")
        print(f"Bandera en {flags}/{args.episodes}")
        venv.close()
        return

    if args.start_col is not None:
        # Evaluar un tramo concreto: un entorno Gymnasium normal, para pasar
        # options={"start_col": ...} en cada reset. Solo con obs=grid (con
        # pixels habría que replicar el apilado de VecFrameStack a mano).
        if args.obs != "grid":
            raise SystemExit("--start-col solo está disponible con --obs grid")
        import gymnasium as gym

        import mario_env  # noqa: F401  registra el entorno

        env = gym.make(mario_env.ENV_ID, obs_type="grid", level=args.level,
                       sticky_prob=args.sticky_prob, goomba_speed_range=goomba_range)
        model = PPO.load(args.model, device="cpu")
        print(f"Modo estadistico: {args.episodes} episodios desde la col. {args.start_col}, "
              f"obs=grid, nivel {args.level}")
        results = []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep, options={"start_col": args.start_col})
            done = False
            steps = 0
            total = 0.0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, term, trunc, info = env.step(int(action))
                total += float(reward)
                steps += 1
                done = term or trunc
            results.append({"progress": float(info.get("progress", 0.0)),
                            "steps": int(info.get("steps", steps)),
                            "flag": bool(info.get("flag_get")),
                            "death_reason": info.get("death_reason"), "reward": total})
        print_report(results)
        env.close()
        return

    # Modo estadístico normal: el mismo VecEnv que el entrenamiento (un entorno),
    # así la observación coincide siempre, también con pixels y frame stack.
    venv = make_mario_venv(
        n_envs=1, seed=args.seed, obs_type=args.obs, level=args.level,
        frame_stack=args.frame_stack, sticky_prob=args.sticky_prob,
        goomba_speed_range=goomba_range, vec="dummy",
    )
    model = PPO.load(args.model, device="cpu")
    print(f"Modo estadistico: {args.episodes} episodios, obs={args.obs}, nivel {args.level}"
          + (f", sticky={args.sticky_prob}" if args.sticky_prob else "")
          + (f", goomba_speed={goomba_range}" if goomba_range else ""))

    results = evaluate_stats(model, venv, args.episodes, args.deterministic, args.seed)
    print_report(results)
    venv.close()


if __name__ == "__main__":
    main()
