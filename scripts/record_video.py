"""record_video.py — graba un vídeo del agente jugando (Persona 4, P4.9).

Carga un modelo PPO y graba uno o varios episodios en un archivo MP4 (o GIF),
usando el render del entorno (`render_mode="rgb_array"`, frame con HUD, de
INTERFACES.md §7). El vídeo demuestra que el agente completa el nivel.

    python scripts/record_video.py --model models/lvl1_grid_s0/final.zip
    python scripts/record_video.py --model models/lvl1_grid_s0/final.zip \
        --episodes 3 --out docs/captures/agente_nivel1.mp4

El MP4 necesita imageio-ffmpeg; si no está, cae a GIF automáticamente. El GIF no
necesita nada extra (imageio viene con pygame/numpy del proyecto... si no, se
avisa de cómo instalarlo).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _frames_de_episodio(model, env, deterministic, seed):
    """Juega un episodio y devuelve la lista de frames (rgb_array) y el info final."""
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, term, trunc, info = env.step(int(action))
        frames.append(env.render())
        done = term or trunc
    return frames, info


def _guardar(frames, out: Path, fps: int) -> Path:
    """Guarda los frames en `out`. MP4 si hay ffmpeg, si no GIF."""
    import imageio

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        try:
            imageio.mimsave(out, frames, fps=fps, macro_block_size=None)
            return out
        except Exception as exc:   # ffmpeg ausente u otro problema del códec
            gif = out.with_suffix(".gif")
            print(f"No se pudo escribir MP4 ({exc}); guardando GIF en {gif}")
            imageio.mimsave(gif, frames, duration=1000 / fps)
            return gif
    imageio.mimsave(out, frames, duration=1000 / fps)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Graba un vídeo del agente completando el nivel (P4.9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="Ruta del .zip del modelo PPO.")
    parser.add_argument("--obs", choices=["grid", "pixels"], default="grid")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1,
                        help="Episodios a grabar (se concatenan en un solo vídeo).")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30, help="Frames por segundo del vídeo.")
    parser.add_argument("--out", default="docs/captures/agente.mp4",
                        help="Archivo de salida (.mp4 o .gif).")
    args = parser.parse_args(argv)

    try:
        import imageio  # noqa: F401
    except ImportError:
        raise SystemExit("Falta imageio. Instálalo con: pip install imageio imageio-ffmpeg")

    import gymnasium as gym
    from stable_baselines3 import PPO

    import mario_env  # noqa: F401  registra el entorno
    from gymnasium.wrappers import FrameStackObservation

    # render_mode="rgb_array": cada env.render() devuelve un frame con HUD.
    env = gym.make(mario_env.ENV_ID, obs_type=args.obs, level=args.level,
                   render_mode="rgb_array")
    if args.frame_stack > 1 and args.obs == "pixels":
        env = FrameStackObservation(env, stack_size=args.frame_stack)

    model = PPO.load(args.model, device="cpu")
    all_frames = []
    for ep in range(args.episodes):
        frames, info = _frames_de_episodio(model, env, args.deterministic, args.seed + ep)
        all_frames.extend(frames)
        print(f"  episodio {ep + 1}: {len(frames)} frames, progreso "
              f"{info.get('progress', 0) * 100:.1f}%, "
              f"{'BANDERA' if info.get('flag_get') else 'muerte:' + str(info.get('death_reason'))}")
    env.close()

    saved = _guardar(all_frames, Path(args.out), args.fps)
    print(f"Vídeo guardado en {saved} ({len(all_frames)} frames a {args.fps} FPS)")


if __name__ == "__main__":
    main()
