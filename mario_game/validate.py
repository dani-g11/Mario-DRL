"""Validador de niveles (Persona 2, P2.9): ¿se puede pasar el nivel desde cada checkpoint?

Un bot sencillo intenta llegar a la bandera desde cada checkpoint del nivel, con el
motor real y a una o varias velocidades de goomba. El informe dice, para cada
checkpoint, si llegó o dónde y por qué murió.

    python -m mario_game.validate                      # nivel 1, informe por pantalla
    python -m mario_game.validate --level 2 --goomba-speeds 0.3 0.5 1.0
    python -m mario_game.validate --report docs/validacion_nivel1.md

Termina con código 0 si el bot gana en todos los casos y 1 si falla alguno.

Cómo leer el resultado: si el bot gana, el tramo es superable (queda demostrado con
una partida real). Si el bot falla, el tramo puede ser imposible o puede ser que el
bot sea demasiado simple; hay que mirar el nivel y probarlo con `play.py`. Este
módulo no usa pygame: sirve también en un servidor sin pantalla.
"""

import argparse
import sys
from dataclasses import dataclass

from mario_game import level as level_module
from mario_game.constants import FPS, GOOMBA_SPEED, TILE, TIME_LIMIT_FRAMES
from mario_game.game import MarioGame
from mario_game.level import TRAMO_ANCHO

MAX_FRAMES = TIME_LIMIT_FRAMES + FPS      # el motor mata a Mario por tiempo antes de llegar aquí


# --- El bot --------------------------------------------------------------------------

class Bot:
    """Corre a la derecha y salta ante paredes, fosos y goombas.

    Solo mira lo que vería un jugador: el nivel justo delante de Mario y los goombas
    despiertos que se acercan. Mantiene el botón de salto mientras sube, para dar el
    salto más alto, y lo suelta antes de volver a saltar (el motor detecta el flanco).
    """

    WALL_LOOKAHEAD = range(4, 40, 8)      # px por delante en los que busca una pared
    PIT_LOOKAHEAD = 4                     # px por delante en los que busca un foso
    GOOMBA_LOOKAHEAD = 56                 # px por delante en los que busca un goomba

    def __init__(self) -> None:
        self._jumping = False

    def act(self, game) -> tuple[bool, bool, bool, bool]:
        """Los botones (left, right, jump, run) para el siguiente frame."""
        mario = game.mario
        jump = False
        if mario.on_ground:
            self._jumping = False
            if self._hazard_ahead(game) and not game.jump_held:
                jump = self._jumping = True
        elif self._jumping and mario.vy < 0:
            jump = True                   # sigue subiendo: mantiene el botón
        return False, True, jump, True

    def _hazard_ahead(self, game) -> bool:
        level, mario = game.level, game.mario
        front = mario.x + mario.w
        body_row = int((mario.y + mario.h / 2) // TILE)
        wall = any(level.is_solid(body_row, int((front + d) // TILE))
                   for d in self.WALL_LOOKAHEAD)
        pit_col = min(int((front + self.PIT_LOOKAHEAD) // TILE), level.width - 1)
        pit = level.surface_row(pit_col) is None
        goomba = any(g.active and not g.squashed
                     and 0 < g.x - front < self.GOOMBA_LOOKAHEAD
                     and abs(g.y - mario.y) < TILE for g in game.goombas)
        return wall or pit or goomba


# --- Una partida ----------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    """Cómo acabó el bot una partida que empezó en `start_col`."""
    start_col: int
    goomba_speed: float
    won: bool
    death_reason: str | None      # None si ganó o si se agotaron los frames
    progress: float
    frames: int
    end_col: int                  # columna donde terminó Mario

    @property
    def seconds(self) -> float:
        return self.frames / FPS

    @property
    def outcome(self) -> str:
        if self.won:
            return "gana"
        return f"muere ({self.death_reason})" if self.death_reason else "sin terminar"


def play(build_level, start_col: int, goomba_speed: float = GOOMBA_SPEED,
         max_frames: int = MAX_FRAMES, bot=None) -> Result:
    """Una partida del bot desde `start_col` en un nivel recién construido."""
    game = MarioGame(build_level(), goomba_speed=goomba_speed)
    game.reset(start_col=start_col)
    bot = bot or Bot()
    while not (game.dead or game.won) and game.frame < max_frames:
        game.step(*bot.act(game))
    return Result(start_col, goomba_speed, game.won, game.death_reason, game.progress,
                  game.frame, int(game.mario.x // TILE))


def validate_level(build_level, goomba_speeds=(GOOMBA_SPEED,),
                   max_frames: int = MAX_FRAMES) -> list[Result]:
    """Una partida por cada checkpoint y velocidad de goomba, en ese orden."""
    checkpoints = build_level().checkpoints
    return [play(build_level, col, speed, max_frames)
            for col in checkpoints for speed in goomba_speeds]


def all_passed(results) -> bool:
    return bool(results) and all(r.won for r in results)


# --- Informe --------------------------------------------------------------------------

def format_report(title: str, build_level, results, goomba_speeds) -> str:
    """El informe en Markdown (se lee bien también en la terminal)."""
    level = build_level()
    by_case = {(r.start_col, r.goomba_speed): r for r in results}
    passed = sum(r.won for r in results)
    lines = [
        f"# Validación de {title}",
        "",
        f"- Nivel de {level.width} columnas ({level.width * TILE} px), bandera en la columna "
        f"{level.flag_col}, {len(level.goomba_cols)} goombas y {len(level.checkpoints)} checkpoints.",
        "- Un bot sencillo (corre a la derecha y salta ante paredes, fosos y goombas) intenta "
        "llegar a la bandera desde cada checkpoint, con el motor real.",
        f"- Velocidades de goomba probadas (px/frame): "
        f"{', '.join(f'{s:g}' for s in goomba_speeds)}.",
        "",
        "| Tramo | Checkpoint | " + " | ".join(f"Goomba {s:g}" for s in goomba_speeds) + " |",
        "|---|---|" + "---|" * len(goomba_speeds),
    ]
    for col in level.checkpoints:
        cells = []
        for speed in goomba_speeds:
            r = by_case[(col, speed)]
            cells.append(f"OK ({r.seconds:.0f} s)" if r.won else
                         f"FALLA: {r.outcome} en la col. {r.end_col}, {r.progress:.0%}")
        lines.append(f"| {col // TRAMO_ANCHO} | col. {col} | " + " | ".join(cells) + " |")
    lines += ["", f"**Resultado: {passed} de {len(results)} partidas ganadas. "
              + ("Todos los tramos son superables desde cada checkpoint.**" if all_passed(results)
                 else "Hay tramos que el bot no supera: hay que revisarlos.**")]
    if not all_passed(results):
        lines.append("")
        lines.append("Si el bot falla, el tramo puede ser imposible o el bot demasiado simple: "
                     "pruébalo a mano con `python play.py --start-col COL`.")
    return "\n".join(lines) + "\n"


# --- Línea de comandos ----------------------------------------------------------------

def load_builder(number: int):
    """`build_level_<number>` de mario_game.level. ValueError si no existe."""
    builder = getattr(level_module, f"build_level_{number}", None)
    if builder is None:
        raise ValueError(f"no existe build_level_{number}() en mario_game/level.py")
    return builder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprueba con un bot que un nivel se puede pasar desde cada checkpoint.")
    parser.add_argument("--level", type=int, default=1, help="número de nivel (por defecto 1)")
    parser.add_argument("--goomba-speeds", type=float, nargs="+", default=[GOOMBA_SPEED],
                        metavar="V", help=f"velocidades de goomba a probar (por defecto {GOOMBA_SPEED})")
    parser.add_argument("--report", metavar="ARCHIVO", help="guarda el informe en este archivo .md")
    args = parser.parse_args(argv)
    if any(speed <= 0 for speed in args.goomba_speeds):
        parser.error("las velocidades de goomba deben ser positivas")
    try:
        builder = load_builder(args.level)
    except ValueError as exc:
        parser.error(str(exc))
    results = validate_level(builder, args.goomba_speeds)
    report = format_report(f"nivel {args.level}", builder, results, args.goomba_speeds)
    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"Informe guardado en {args.report}")
    return 0 if all_passed(results) else 1


if __name__ == "__main__":
    sys.exit(main())
