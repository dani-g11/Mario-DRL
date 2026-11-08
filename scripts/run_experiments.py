"""run_experiments.py — experimentos, robustez y generalización (P4.7, P4.8).

Orquesta los entrenamientos y las evaluaciones del plan y junta los resultados
en una tabla Markdown (docs/resultados.md por defecto):

- P4.7 Experimentos: grid frente a pixels, con y sin inicio aleatorio, varias
  semillas. Cada combinación es un entrenamiento de `--steps` pasos.
- P4.8 Robustez y generalización: cada modelo entrenado se evalúa en varias
  condiciones (sin aleatoriedad, con acciones pegajosas, con velocidad de goomba
  variable y en el nivel 2), 20 episodios por condición.

Llama a train.py y evaluate.py como subprocesos, así usa exactamente el mismo
código que a mano. Es largo: pensado para lanzarlo una vez y dejarlo correr. Con
`--dry-run` imprime los comandos sin ejecutarlos, para revisar el plan.

    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --steps 2000000 --seeds 0 1 2
    python scripts/run_experiments.py --steps 50000 --seeds 0 --quick   # prueba rápida

El resumen se parsea de la salida de evaluate.py (la línea "Tasa de exito").
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MODELS_DIR = ROOT / "models"


def _env():
    e = os.environ.copy()
    e.setdefault("SDL_VIDEODRIVER", "dummy")
    e.setdefault("SDL_AUDIODRIVER", "dummy")
    return e


def _run(cmd, dry_run) -> str:
    print(">", " ".join(str(c) for c in cmd))
    if dry_run:
        return ""
    result = subprocess.run(cmd, cwd=ROOT, env=_env(), capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"Falló: {' '.join(str(c) for c in cmd)}")
    return result.stdout


# "Tasa de exito:    18/20 = 90.0% ..." -> (18, 20, 90.0)
_RATE_RE = re.compile(r"Tasa de exito:\s+(\d+)/(\d+)\s*=\s*([\d.]+)%")
_PROGRESS_RE = re.compile(r"Progreso medio:\s+([\d.]+)%")


def _parse_eval(output):
    rate = _RATE_RE.search(output)
    prog = _PROGRESS_RE.search(output)
    success = float(rate.group(3)) if rate else None
    progress = float(prog.group(1)) if prog else None
    return success, progress


def train_one(obs, seed, steps, n_envs, random_start, quick, dry_run) -> str:
    name_parts = ["lvl1", obs, f"s{seed}"]
    if random_start:
        name_parts.append("rnd")
    name = "_".join(name_parts)
    cmd = [PY, "train.py", "--obs", obs, "--seed", str(seed), "--steps", str(steps),
           "--n-envs", str(n_envs), "--name", name]
    if obs == "pixels":
        cmd += ["--frame-stack", "4"]
    if random_start:
        cmd += ["--random-start"]
    if quick:
        cmd += ["--vec", "dummy"]
    _run(cmd, dry_run)
    return name


def eval_condition(name, obs, label, extra_args, episodes, dry_run):
    model = f"models/{name}/final.zip"
    cmd = [PY, "evaluate.py", "--model", model, "--obs", obs,
           "--episodes", str(episodes)] + extra_args
    out = _run(cmd, dry_run)
    success, progress = _parse_eval(out)
    return {"model": name, "obs": obs, "condicion": label,
            "exito": success, "progreso": progress}


def conditions(obs, episodes):
    """Las condiciones de evaluación de P4.8 (como listas de args de evaluate.py)."""
    return [
        ("sin aleatoriedad (nivel 1)", []),
        ("acciones pegajosas 0.1", ["--sticky-prob", "0.1"]),
        ("goomba 0.3-1.5", ["--goomba-speed-min", "0.3", "--goomba-speed-max", "1.5"]),
        ("generalización (nivel 2)", ["--level", "2"]),
    ]


def write_report(rows, out_path: Path, steps, seeds) -> None:
    lines = [
        "# Resultados de entrenamiento (Persona 4)",
        "",
        f"- Pasos de entrenamiento por modelo: {steps:,}.",
        f"- Semillas: {', '.join(str(s) for s in seeds)}.",
        "- Evaluación: 20 episodios por condición (criterio del plan: ≥ 90%).",
        "",
        "## Experimentos (P4.7) y robustez/generalización (P4.8)",
        "",
        "| Modelo | Obs | Condición | Tasa de éxito | Progreso medio |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        exito = f"{r['exito']:.1f}%" if r["exito"] is not None else "—"
        prog = f"{r['progreso']:.1f}%" if r["progreso"] is not None else "—"
        lines.append(f"| {r['model']} | {r['obs']} | {r['condicion']} | {exito} | {prog} |")
    lines += ["", "Generado por `scripts/run_experiments.py`.", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nInforme escrito en {out_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Experimentos y evaluaciones de P4.7/P4.8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Pasos por entrenamiento (P4.6: 2 M).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Semillas (P4.7: 3 semillas).")
    parser.add_argument("--obs", nargs="+", default=["grid", "pixels"],
                        choices=["grid", "pixels"], help="Tipos de observación a comparar.")
    parser.add_argument("--random-start", action="store_true",
                        help="Añadir la variante con inicio aleatorio (P4.7).")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20, help="Episodios de evaluación.")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido (vec=dummy), para una prueba con pocos pasos.")
    parser.add_argument("--report", default="docs/resultados.md",
                        help="Archivo del informe de resultados.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprimir los comandos sin ejecutarlos.")
    args = parser.parse_args(argv)

    rows = []
    variants = [False]
    if args.random_start:
        variants.append(True)

    for obs in args.obs:
        for seed in args.seeds:
            for rnd in variants:
                name = train_one(obs, seed, args.steps, args.n_envs, rnd,
                                 args.quick, args.dry_run)
                for label, extra in conditions(obs, args.episodes):
                    rows.append(eval_condition(name, obs, label, extra,
                                              args.episodes, args.dry_run))

    if not args.dry_run:
        write_report(rows, ROOT / args.report, args.steps, args.seeds)


if __name__ == "__main__":
    main()
