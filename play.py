"""Modo humano de Mario DRL (Persona 2, P2.8): juega el nivel con el teclado.

    python play.py                  # nivel 1
    python play.py --scale 2        # ventana más pequeña (por defecto se elige sola)
    python play.py --start-col 94   # empezar en una columna concreta, para probar un tramo

Controles:
    FLECHAS o A D        moverse
    ESPACIO, ARRIBA o W  saltar (mantenerlo da un salto más alto)
    SHIFT o X            correr
    R  o  ENTER          reiniciar (ENTER solo al terminar la partida)
    1-9                  reiniciar desde el checkpoint n del nivel (0: desde el inicio)
    P                    pausa
    ESC o Q              salir

El juego avanza a 60 FPS con el mismo motor que usa el agente, así que sirve para
comprobar a mano que el nivel se puede pasar y para grabar el vídeo del informe.
"""

import argparse
import sys

import pygame

from mario_game.constants import FPS, GOOMBA_SPEED, SCREEN_H, SCREEN_W
from mario_game.game import MarioGame
from mario_game.level import build_level_1
from mario_game.render import draw_banner

LEVELS = {1: build_level_1}                 # número de nivel -> función que lo construye
CONTROLS = "Controles:" + __doc__.split("Controles:")[1].split("El juego")[0].rstrip()

DEFAULT_SCALE = 3                           # 768×720: cabe en una pantalla de 1080 px
WINDOW_MARGIN = 120                         # alto que se reserva para la barra de título y la de tareas

KEYS_LEFT = (pygame.K_LEFT, pygame.K_a)
KEYS_RIGHT = (pygame.K_RIGHT, pygame.K_d)
KEYS_JUMP = (pygame.K_SPACE, pygame.K_UP, pygame.K_w)
KEYS_RUN = (pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_x)
KEYS_QUIT = (pygame.K_ESCAPE, pygame.K_q)
DIGIT_KEYS = {pygame.K_0 + n: n for n in range(10)}

DEATH_TEXT = {"pit": "CAISTE EN UN FOSO", "goomba": "TE TOCO UN GOOMBA",
              "time": "SE ACABO EL TIEMPO"}      # sin tildes: la fuente del render no las tiene
RED = (252, 84, 84)
GREEN = (88, 216, 84)


# --- Entrada y ventana --------------------------------------------------------------

def read_inputs(pressed) -> tuple[bool, bool, bool, bool]:
    """(left, right, jump, run) a partir del estado del teclado (`pygame.key.get_pressed()`)."""
    def down(keys):
        return any(pressed[key] for key in keys)
    return down(KEYS_LEFT), down(KEYS_RIGHT), down(KEYS_JUMP), down(KEYS_RUN)


def choose_scale(requested: int | None, desktop_size: tuple[int, int] | None) -> int:
    """Escala entera de la ventana.

    Si el usuario la pide, se respeta. Si no, la mayor (hasta DEFAULT_SCALE) con la que
    la ventana cabe en el escritorio; si no se conoce su tamaño, DEFAULT_SCALE.
    """
    if requested is not None:
        return max(1, requested)
    if not desktop_size or min(desktop_size) <= 0:
        return DEFAULT_SCALE
    width, height = desktop_size
    fits = min(width // SCREEN_W, (height - WINDOW_MARGIN) // SCREEN_H)
    return max(1, min(DEFAULT_SCALE, fits))


# --- La partida ----------------------------------------------------------------------

class Session:
    """Una sesión de juego: el motor, la pausa y el punto de inicio elegido.

    No toca la ventana ni el teclado (recibe el estado de las teclas y una superficie),
    así que se prueba sin pantalla.
    """

    def __init__(self, level_number: int = 1, start_col: int | None = None,
                 goomba_speed: float = GOOMBA_SPEED, hud: bool = True) -> None:
        if level_number not in LEVELS:
            raise ValueError(f"el nivel {level_number} no existe (hay: {sorted(LEVELS)})")
        self.level_number = level_number
        self.start_col = start_col          # None: el inicio del nivel
        self.hud = hud
        self.game = MarioGame(LEVELS[level_number](), goomba_speed=goomba_speed)
        self.game.reset(start_col)          # ValueError si start_col es un foso o no existe
        self.paused = False
        self.episode = 1
        self._reported = False              # ¿ya se contó el final de este episodio?

    # Controles ------------------------------------------------------------------

    def restart(self) -> None:
        self.game.reset(self.start_col)
        self.paused = False
        self.episode += 1
        self._reported = False

    def jump_to_checkpoint(self, number: int) -> bool:
        """Reinicia desde el checkpoint `number` (1 = el primero); 0 vuelve al inicio del nivel."""
        checkpoints = self.game.level.checkpoints
        if number == 0:
            self.start_col = None
        elif 1 <= number <= len(checkpoints):
            self.start_col = checkpoints[number - 1]
        else:
            return False
        self.restart()
        return True

    def toggle_pause(self) -> None:
        if not self.finished:
            self.paused = not self.paused

    def handle_key(self, key: int) -> None:
        """Reacciona a una tecla pulsada (no a las de movimiento, que se leen en `update`)."""
        if key == pygame.K_r or (key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.finished):
            self.restart()
        elif key == pygame.K_p:
            self.toggle_pause()
        elif key in DIGIT_KEYS:
            self.jump_to_checkpoint(DIGIT_KEYS[key])

    # Un frame -------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.game.dead or self.game.won

    def update(self, pressed) -> str | None:
        """Avanza un frame con las teclas pulsadas.

        Devuelve el resumen del episodio una sola vez, en el frame en que termina.
        """
        if not self.paused:
            self.game.step(*read_inputs(pressed))
        if self.finished and not self._reported:
            self._reported = True
            return self.summary()
        return None

    def summary(self) -> str:
        game = self.game
        outcome = "victoria" if game.won else f"derrota ({game.death_reason})"
        seconds = max(game.time_left, 0) // FPS
        return (f"Episodio {self.episode}: {outcome} | progreso {int(game.progress * 100)}% | "
                f"puntos {game.score} | monedas {game.coins} | tiempo restante {seconds} s")

    # Dibujo ---------------------------------------------------------------------

    def banner(self) -> tuple[str, list[str], tuple] | None:
        """(título, líneas, color) del cartel que toca mostrar, o None si se está jugando."""
        game = self.game
        if game.won:
            return "NIVEL COMPLETADO", [f"PUNTOS {game.score:06d}", "R REINICIAR  ESC SALIR"], GREEN
        if game.dead:
            reason = DEATH_TEXT.get(game.death_reason, "")
            return "GAME OVER", [reason, "R REINICIAR  ESC SALIR"], RED
        if self.paused:
            return "PAUSA", ["P CONTINUAR"], (252, 252, 252)
        return None

    def draw(self, surface) -> None:
        """Dibuja el juego (y el cartel, si toca) en una superficie de SCREEN_W × SCREEN_H."""
        self.game.render(surface, hud=self.hud)
        banner = self.banner()
        if banner:
            title, lines, color = banner
            draw_banner(surface, title, lines, color)


# --- Bucle principal -----------------------------------------------------------------

def handle_event(session: Session, event) -> bool:
    """Procesa un evento de pygame. Devuelve False si hay que salir."""
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.KEYDOWN:
        if event.key in KEYS_QUIT:
            return False
        session.handle_key(event.key)
    return True


def run(session: Session, scale: int | None = None, max_frames: int | None = None) -> None:
    """Abre la ventana y juega hasta que el usuario sale (o hasta `max_frames`, para pruebas)."""
    pygame.display.init()
    try:
        desktop = pygame.display.get_desktop_sizes()[0]
    except (pygame.error, IndexError):
        desktop = None
    scale = choose_scale(scale, desktop)
    window = pygame.display.set_mode((SCREEN_W * scale, SCREEN_H * scale))
    pygame.display.set_caption("Mario DRL")
    canvas = pygame.Surface((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    frames, running = 0, True
    try:
        while running and (max_frames is None or frames < max_frames):
            for event in pygame.event.get():
                running = handle_event(session, event) and running
            summary = session.update(pygame.key.get_pressed())
            if summary:
                print(summary)
            session.draw(canvas)
            pygame.transform.scale(canvas, window.get_size(), window)    # vecino más cercano: píxel nítido
            pygame.display.flip()
            clock.tick(FPS)
            frames += 1
    finally:
        pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Juega Mario DRL con el teclado.",
                                     epilog=CONTROLS,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, default=1, choices=sorted(LEVELS),
                        help="nivel a jugar (por defecto 1)")
    parser.add_argument("--scale", type=int, default=None,
                        help="ampliación de la ventana (por defecto la mayor que cabe, hasta 3)")
    parser.add_argument("--start-col", type=int, default=None, metavar="COL",
                        help="columna de inicio (por defecto la del nivel)")
    parser.add_argument("--goomba-speed", type=float, default=GOOMBA_SPEED, metavar="V",
                        help=f"velocidad de los goombas en px/frame (por defecto {GOOMBA_SPEED})")
    parser.add_argument("--no-hud", action="store_true", help="oculta el HUD")
    parser.add_argument("--frames", type=int, default=None, metavar="N",
                        help="sale solo tras N frames (para pruebas)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scale is not None and args.scale < 1:
        parser.error("--scale debe ser 1 o más")
    try:
        session = Session(args.level, args.start_col, args.goomba_speed, hud=not args.no_hud)
    except ValueError as exc:
        parser.error(str(exc))
    print(CONTROLS)
    run(session, args.scale, args.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
