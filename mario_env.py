"""MarioEnv: entorno Gymnasium para el juego Mario DRL (Persona 3).

Contrato en INTERFACES.md §7: 7 acciones (Discrete(7)) mantenidas 4 frames
(frame skip), observación, recompensa, terminación/truncado e `info`.

pygame no se importa arriba del archivo: solo lo necesitan la observación
`pixels` y el render, así que se importa dentro de esos métodos. El modo `grid`
no carga pygame.

Observación `grid` (P3.3): un vector float32 de GRID_OBS_DIM valores en [-1, 1],
con estas partes en este orden:

    sólidos    ROWS × GRID_COLS   1 si el tile es sólido (fuera del mapa por los
                                  lados también). Ventana de GRID_COLS_BEHIND
                                  columnas por detrás de Mario y GRID_COLS_AHEAD
                                  por delante.
    enemigos   ROWS × GRID_COLS   1 en la celda del centro de cada goomba activo
                                  y no aplastado.
    estado     STATE_DIM          y/SCREEN_H, vx/RUN_MAX_SPEED, vy/TILE,
                                  on_ground, fracción de tile de Mario en x,
                                  jump_held.
    cercanos   N_ENEMIES_OBS × 4  para los goombas más cercanos por delante:
                                  dx/SCREEN_W, dy/SCREEN_H, direction, 1
                                  (presente). Ceros si no hay.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mario_game.constants import (FPS, ROWS, RUN_MAX_SPEED, SCREEN_H, SCREEN_W,
                                  SOLID_TILES, TILE)

# (left, right, jump, run): orden e índices de INTERFACES.md §7.
ACTIONS = [
    (False, False, False, False),  # 0 nada
    (False, True,  False, False),  # 1 →
    (False, True,  True,  False),  # 2 → + saltar
    (False, True,  False, True),   # 3 → + correr
    (False, True,  True,  True),   # 4 → + correr + saltar
    (False, False, True,  False),  # 5 saltar
    (True,  False, False, False),  # 6 ←
]
ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+JUMP", "RIGHT+RUN", "RIGHT+RUN+JUMP",
                "JUMP", "LEFT"]

OBS_TYPES = ("grid",)

# --- Observación grid --------------------------------------------------------
GRID_COLS_BEHIND = 3
GRID_COLS_AHEAD = 12
GRID_COLS = GRID_COLS_BEHIND + 1 + GRID_COLS_AHEAD     # 16
STATE_DIM = 6
N_ENEMIES_OBS = 3
ENEMY_FEATURES = 4
GRID_CELLS = ROWS * GRID_COLS                          # 240 por canal
STATE_OFFSET = 2 * GRID_CELLS                          # dónde empieza el estado
ENEMIES_OFFSET = STATE_OFFSET + STATE_DIM              # dónde empiezan los cercanos
GRID_OBS_DIM = ENEMIES_OFFSET + N_ENEMIES_OBS * ENEMY_FEATURES   # 498

# Un goomba cuenta como "cercano" si está en esta franja de dx (en pantallas):
# los que van delante importan más que los que ya quedaron atrás.
NEAR_DX_MIN = -0.5
NEAR_DX_MAX = 1.0

# Tabla código de tile -> 1.0 si es sólido. Con ella la máscara del nivel entero
# se calcula de una vez con numpy en vez de recorrer la lista de listas.
_SOLID_LUT = np.zeros(256, dtype=np.float32)
_SOLID_LUT[list(SOLID_TILES)] = 1.0


def _default_factory():
    """Fábrica por defecto. En P3.8 pasará al motor real."""
    from mario_stub import StubMarioGame
    return StubMarioGame()


class MarioEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": FPS}

    def __init__(self, game_factory=None, obs_type="grid", frame_skip=4):
        super().__init__()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"obs_type={obs_type!r} no existe (hay: {OBS_TYPES})")
        self.obs_type = obs_type
        self.frame_skip = frame_skip
        self._game_factory = game_factory if game_factory is not None else _default_factory
        self.game = self._game_factory()

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, (GRID_OBS_DIM,), np.float32)
        self._build_solid_mask()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self._build_solid_mask()
        return self._obs(), {}

    def step(self, action):
        left, right, jump, run = ACTIONS[int(action)]
        for _ in range(self.frame_skip):
            self.game.step(left, right, jump, run)
            if self.game.dead or self.game.won:
                break
        terminated = self.game.dead or self.game.won
        return self._obs(), 0.0, terminated, False, {}

    # --- Observación -----------------------------------------------------------

    def _obs(self):
        return self._obs_grid()

    def _build_solid_mask(self) -> None:
        """Máscara de sólidos del nivel entero, con GRID_COLS columnas sólidas a
        cada lado (fuera del mapa por los lados es sólido, INTERFACES.md §3).

        Se hace una vez por reset(): level.tiles es una lista de listas y
        recorrerla en cada paso sería lento. Sigue siendo válida aunque un
        bloque ? pase a usado durante el episodio, porque los dos son sólidos.
        La columna c del nivel es la columna c + GRID_COLS de la máscara.
        """
        tiles = np.asarray(self.game.level.tiles, dtype=np.uint8)
        pad = np.ones((ROWS, GRID_COLS), dtype=np.float32)
        self._solid = np.concatenate([pad, _SOLID_LUT[tiles], pad], axis=1)

    def _obs_grid(self) -> np.ndarray:
        game = self.game
        m = game.mario
        c0 = int((m.x + m.w / 2) // TILE) - GRID_COLS_BEHIND   # primera columna de la ventana
        solid = self._solid[:, c0 + GRID_COLS: c0 + 2 * GRID_COLS]

        enemy = np.zeros((ROWS, GRID_COLS), dtype=np.float32)
        near = []
        for g in game.goombas:
            if g.squashed or not g.active:
                continue                 # aplastados y dormidos no cuentan (§4)
            col = int((g.x + g.w / 2) // TILE) - c0
            row = int((g.y + g.h / 2) // TILE)
            if 0 <= col < GRID_COLS and 0 <= row < ROWS:
                enemy[row, col] = 1.0
            dx = (g.x - m.x) / SCREEN_W
            if NEAR_DX_MIN < dx < NEAR_DX_MAX:
                near.append((abs(dx), dx, (g.y - m.y) / SCREEN_H, g.direction))
        near.sort(key=lambda e: e[0])

        # Solo constantes del contrato: |vx| <= RUN_MAX_SPEED y |vy| < TILE (§5).
        # y se recorta: justo antes de morir en un foso Mario está bajo el mapa.
        extra = np.zeros(STATE_DIM + N_ENEMIES_OBS * ENEMY_FEATURES, dtype=np.float32)
        extra[0] = min(max(m.y / SCREEN_H, -1.0), 1.0)
        extra[1] = m.vx / RUN_MAX_SPEED
        extra[2] = m.vy / TILE
        extra[3] = float(m.on_ground)
        extra[4] = ((m.x + m.w / 2) % TILE) / TILE
        extra[5] = float(game.jump_held)
        for i, (_, dx, dy, direction) in enumerate(near[:N_ENEMIES_OBS]):
            j = STATE_DIM + i * ENEMY_FEATURES
            extra[j:j + ENEMY_FEATURES] = (dx, min(max(dy, -1.0), 1.0), direction, 1.0)

        return np.concatenate([solid.ravel(), enemy.ravel(), extra])
