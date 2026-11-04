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

Observación `pixels` (P3.4): la vista del juego sin HUD, reducida a
PIXELS_SIZE × PIXELS_SIZE y en grises, uint8 de forma (84, 84, 1). El apilado
de frames lo hace quien entrena (VecFrameStack, P4).

Recompensa y fin de episodio (P3.5, valores del plan de trabajo):

    recompensa  Δmario.x / 10 − 0.01 por paso; −15 al morir; +50 en la bandera.
                El avance es Δmario.x (no Δmax_x): retroceder resta, y la suma de
                un episodio es la distancia neta recorrida / 10.
    terminated  Mario muere (foso, goomba o tiempo) o toca la bandera.
    truncated   sin terminar, `stuck_steps` pasos seguidos sin superar en más de
                1 px el mejor x del episodio, o `max_steps` pasos en total.

`info` (P3.6): x_pos, progress, flag_get, dead y death_reason (contrato), más
score, coins, time_left, steps y start_col. El entorno se registra como
"MarioPygame-v0".

Inicio aleatorio (P3.7): con random_start=True, cada reset() empieza con
probabilidad random_start_prob en un checkpoint del nivel elegido con
self.np_random. options={"start_col": c} tiene prioridad.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mario_game.constants import (FPS, ROWS, RUN_MAX_SPEED, SCREEN_H, SCREEN_W,
                                  SOLID_TILES, TILE)
from mario_game.game import MarioGame
from mario_game.level import build_level_1

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

OBS_TYPES = ("grid", "pixels")

# --- Recompensa y fin de episodio (P3.5) -------------------------------------
PROGRESS_REWARD_DIV = 10.0     # avance en x / 10
STEP_PENALTY = 0.01            # por paso, para que no le salga gratis quedarse quieto
DEATH_PENALTY = 15.0
FLAG_REWARD = 50.0
STUCK_STEPS = 150              # pasos sin avanzar antes de truncar
MAX_STEPS = 3000               # 12 000 frames (200 s): se trunca antes del límite de tiempo
PROGRESS_MIN_PX = 1.0          # cuánto hay que superar el mejor x para contar como avance

# --- Observación pixels ------------------------------------------------------
PIXELS_SIZE = 84
# Pesos de luminancia (ITU-R BT.601) para pasar de RGB a grises.
_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

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
    """Fábrica por defecto (P3.8): el motor real con el nivel 1.

    Crea un Level nuevo en cada llamada ("Un Level por MarioGame",
    INTERFACES.md §4): el motor modifica los bloques ?, y con varios entornos
    en paralelo cada uno necesita el suyo. El stub (mario_stub.StubMarioGame)
    se queda para los tests: se pasa con game_factory=StubMarioGame.
    """
    return MarioGame(build_level_1())


class MarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, game_factory=None, obs_type="grid", frame_skip=4,
                 stuck_steps=STUCK_STEPS, max_steps=MAX_STEPS,
                 random_start=False, random_start_prob=0.5,
                 render_mode=None, render_scale=3):
        super().__init__()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"obs_type={obs_type!r} no existe (hay: {OBS_TYPES})")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode={render_mode!r} no existe "
                             f"(hay: {self.metadata['render_modes']})")
        if not 0.0 <= random_start_prob <= 1.0:
            raise ValueError(f"random_start_prob={random_start_prob} debe estar en [0, 1]")
        self.obs_type = obs_type
        self.frame_skip = frame_skip
        self.stuck_steps = stuck_steps
        self.max_steps = max_steps
        self.random_start = random_start
        self.random_start_prob = random_start_prob
        self.render_mode = render_mode
        self.render_scale = render_scale
        self._game_factory = game_factory if game_factory is not None else _default_factory
        self.game = self._game_factory()

        self.action_space = spaces.Discrete(len(ACTIONS))
        if obs_type == "grid":
            self.observation_space = spaces.Box(-1.0, 1.0, (GRID_OBS_DIM,), np.float32)
        else:
            self.observation_space = spaces.Box(0, 255, (PIXELS_SIZE, PIXELS_SIZE, 1),
                                                np.uint8)
        self._start_col = self.game.level.start_col
        self._build_solid_mask()
        self._reset_counters()
        self._canvas = None              # Surface de SCREEN_W × SCREEN_H, se crea al usarla
        self._window = None              # ventana del modo "human"
        self._clock = None

    def reset(self, *, seed=None, options=None):
        """Empieza un episodio.

        Columna de inicio (P3.7), por orden de prioridad:
        1. options={"start_col": c}: esa columna (evaluación y tests).
        2. random_start: con probabilidad random_start_prob, un checkpoint del
           nivel al azar, para practicar los tramos finales sin tener que
           llegar hasta ellos.
        3. Si no, level.start_col.
        """
        super().reset(seed=seed)
        self._start_col = self._choose_start_col(options)
        self.game.reset(self._start_col)
        self._build_solid_mask()
        self._reset_counters()
        if self.render_mode == "human":
            self._render_human()
        return self._obs(), self._info()

    def _choose_start_col(self, options) -> int:
        level = self.game.level
        if options and options.get("start_col") is not None:
            return int(options["start_col"])
        # Siempre self.np_random (nunca random ni np.random globales): así la
        # misma semilla da el mismo inicio. Con random_start=False no se consume
        # ningún número aleatorio.
        if self.random_start and self.np_random.random() < self.random_start_prob:
            # int(): choice devuelve numpy.int64; el motor y el info reciben un int.
            return int(self.np_random.choice(level.checkpoints))
        return level.start_col

    def step(self, action):
        game = self.game
        left, right, jump, run = ACTIONS[int(action)]
        prev_x = game.mario.x
        for _ in range(self.frame_skip):
            game.step(left, right, jump, run)
            if self.render_mode == "human":
                self._render_human()     # cada frame, para que se vea fluido
            if game.dead or game.won:
                break

        reward = (game.mario.x - prev_x) / PROGRESS_REWARD_DIV - STEP_PENALTY
        if game.dead:
            reward -= DEATH_PENALTY
        if game.won:
            reward += FLAG_REWARD

        self._steps += 1
        if game.mario.x > self._best_x + PROGRESS_MIN_PX:
            self._best_x = game.mario.x
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1

        terminated = bool(game.dead or game.won)
        truncated = (not terminated) and (
            self._steps_without_progress >= self.stuck_steps
            or self._steps >= self.max_steps)
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _info(self) -> dict:
        """Diccionario `info` de reset() y step().

        Las 5 primeras claves son las del contrato (INTERFACES.md §7) y están
        siempre. Las demás son extras para el HUD y las métricas de P4.
        """
        game = self.game
        return {
            "x_pos": float(game.mario.x),
            "progress": float(game.progress),
            "flag_get": bool(game.won),
            "dead": bool(game.dead),
            "death_reason": game.death_reason,
            "score": int(game.score),
            "coins": int(game.coins),
            "time_left": int(game.time_left),
            "steps": self._steps,
            "start_col": self._start_col,
        }

    def _reset_counters(self) -> None:
        """Contadores del episodio para el truncado (P3.5)."""
        self._steps = 0
        self._steps_without_progress = 0
        self._best_x = self.game.mario.x

    def render(self):
        """"rgb_array": el frame actual con HUD, (SCREEN_H, SCREEN_W, 3) uint8.
        "human": la ventana ya se actualiza en reset() y step()."""
        if self.render_mode == "rgb_array":
            import pygame

            canvas = self._get_canvas()
            self.game.render(canvas, hud=True)
            # surfarray devuelve (ancho, alto, 3); Gymnasium espera (alto, ancho, 3).
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        if self.render_mode == "human":
            self._render_human()
        return None

    def close(self):
        if self._window is not None:
            import pygame

            pygame.display.quit()
            self._window = None
            self._clock = None

    # --- Observación -----------------------------------------------------------

    def _obs(self):
        if self.obs_type == "pixels":
            return self._obs_pixels()
        return self._obs_grid()

    def _obs_pixels(self) -> np.ndarray:
        """La vista sin HUD (los números del marcador son ruido para la red),
        reducida a 84 × 84 y en grises. render.py no usa pygame.display, así que
        funciona sin ventana."""
        import pygame

        canvas = self._get_canvas()
        self.game.render(canvas, hud=False)
        small = pygame.transform.smoothscale(canvas, (PIXELS_SIZE, PIXELS_SIZE))
        rgb = pygame.surfarray.array3d(small).astype(np.float32)   # ejes (x, y, color)
        gray = rgb @ _GRAY_WEIGHTS                                 # (x, y)
        return gray.T.astype(np.uint8)[..., None]                  # (y, x, 1)

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

    # --- Render ----------------------------------------------------------------

    def _get_canvas(self):
        """Una sola Surface reutilizada en cada paso (crearla cada vez es lento)."""
        if self._canvas is None:
            import pygame

            self._canvas = pygame.Surface((SCREEN_W, SCREEN_H))
        return self._canvas

    def _render_human(self) -> None:
        """Dibuja el frame en una ventana escalada, a FPS frames por segundo.

        Si se cierra la ventana, el entorno sigue funcionando sin render (así no
        se corta un entrenamiento o una evaluación a medias).
        """
        import pygame

        if self._window is None:
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (SCREEN_W * self.render_scale, SCREEN_H * self.render_scale))
            pygame.display.set_caption("Mario DRL - agente")
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                self.render_mode = None
                return
        canvas = self._get_canvas()
        self.game.render(canvas, hud=True)
        pygame.transform.scale(canvas, self._window.get_size(), self._window)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])


# Registro en Gymnasium (INTERFACES.md §7). Ocurre al importar este módulo: hay
# que hacer `import mario_env` antes de gym.make(ENV_ID). Los argumentos de
# MarioEnv pasan por gym.make, p. ej. gym.make(ENV_ID, obs_type="pixels").
ENV_ID = "MarioPygame-v0"
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="mario_env:MarioEnv")
