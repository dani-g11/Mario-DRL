"""StubMarioGame: un juego falso que cumple el contrato (INTERFACES.md §8).

Mario avanza en línea recta, sin física ni colisiones. Sirve para probar el
entorno (mario_env.py) sin depender del motor real: los tests del entorno que
necesitan números exactos usan este stub porque es determinista.

Reutiliza mario_game.game.Mario como contenedor de datos para no reescribir
x, y, w, h, vx, vy, on_ground, facing ni las propiedades right/bottom.
"""

from mario_game.constants import (GOOMBA_SPEED, MARIO_H, MARIO_W, RUN_MAX_SPEED,
                                   SCREEN_W, TILE, TIME_LIMIT_FRAMES,
                                   WALK_MAX_SPEED)
from mario_game.game import Mario
from mario_game.level import build_level_1


class StubMarioGame:
    """Doble de prueba de MarioGame. Avanza en línea recta, sin física."""

    def __init__(self, level=None, goomba_speed=GOOMBA_SPEED,
                 time_limit=TIME_LIMIT_FRAMES):
        self.level = build_level_1() if level is None else level
        self.goomba_speed = goomba_speed
        self._time_limit = time_limit
        # _max_cam_x como en el motor: el borde derecho de la vista no pasa del
        # final del nivel; si el nivel es más estrecho que la pantalla, 0.
        self._max_cam_x = float(max(0, self.level.width * TILE - SCREEN_W))
        self.reset()

    def reset(self, start_col=None) -> None:
        level = self.level
        col = level.start_col if start_col is None else start_col
        row = level.surface_row(col)
        if row is None:
            raise ValueError(f"start_col {col} cae sobre un foso")

        # Mario colocado como el motor: centrado en la columna, pies en el suelo.
        self.mario = Mario(col * TILE + (TILE - MARIO_W) / 2, row * TILE - MARIO_H)
        self.mario.on_ground = True

        self.goombas = []
        self.dead = False
        self.won = False
        self.death_reason = None
        self.jump_held = False
        self.score = 0
        self.coins = 0
        self.time_left = self._time_limit
        self.frame = 0

        # cam_x y progress como el motor: con un start_col lejano no valen 0.
        self._max_x = self.mario.x
        self.cam_x = 0.0
        self._update_camera()
        self.progress = 0.0
        self._update_progress()

    def step(self, left, right, jump, run) -> None:
        if self.dead or self.won:
            return                       # estado congelado hasta reset()

        m = self.mario
        if right:
            m.vx = RUN_MAX_SPEED if run else WALK_MAX_SPEED
            m.x += m.vx
            m.facing = 1
        else:
            m.vx = 0.0

        self._update_camera()
        self._update_progress()

        if m.right > self.level.flag_col * TILE:
            self.won = True
            self.progress = 1.0

        self.time_left -= 1
        if self.time_left <= 0 and not (self.dead or self.won):
            self.dead = True
            self.death_reason = "time"

        self.jump_held = jump
        self.frame += 1

    def render(self, surface, hud: bool = True) -> None:
        # pygame se importa aquí dentro para que el stub funcione sin pygame,
        # igual que el motor real.
        import pygame

        surface.fill((92, 148, 252))
        pygame.draw.rect(
            surface, (228, 52, 52),
            pygame.Rect(int(self.mario.x - self.cam_x), int(self.mario.y),
                        self.mario.w, self.mario.h),
        )

    # --- Igual que el motor: cámara que no retrocede y progreso que no baja ---

    def _update_camera(self) -> None:
        target = self.mario.x + self.mario.w / 2 - SCREEN_W / 2
        self.cam_x = min(max(self.cam_x, target, 0.0), self._max_cam_x)

    def _update_progress(self) -> None:
        self._max_x = max(self._max_x, self.mario.x)
        if self.won:
            self.progress = 1.0
        else:
            flag_x = self.level.flag_col * TILE
            self.progress = min(self._max_x / flag_x, 1.0) if flag_x > 0 else 1.0
