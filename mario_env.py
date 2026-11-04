"""MarioEnv: entorno Gymnasium para el juego Mario DRL (Persona 3).

Contrato en INTERFACES.md §7: 7 acciones (Discrete(7)) mantenidas 4 frames
(frame skip), observación, recompensa, terminación/truncado e `info`.

pygame no se importa arriba del archivo: solo lo necesitan la observación
`pixels` y el render, así que se importa dentro de esos métodos. El modo `grid`
no carga pygame.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mario_game.constants import FPS, RUN_MAX_SPEED, TILE

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


def _default_factory():
    """Fábrica por defecto. En P3.8 pasará al motor real."""
    from mario_stub import StubMarioGame
    return StubMarioGame()


class MarioEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": FPS}

    def __init__(self, game_factory=None, frame_skip=4):
        super().__init__()
        self.frame_skip = frame_skip
        self._game_factory = game_factory if game_factory is not None else _default_factory
        self.game = self._game_factory()

        self.action_space = spaces.Discrete(len(ACTIONS))
        # Observación provisional (se sustituye en P3.3).
        self.observation_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def _obs(self):
        m = self.game.mario
        width_px = self.game.level.width * TILE
        return np.array([m.x / width_px, m.vx / RUN_MAX_SPEED],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._obs(), {}

    def step(self, action):
        left, right, jump, run = ACTIONS[int(action)]
        for _ in range(self.frame_skip):
            self.game.step(left, right, jump, run)
            if self.game.dead or self.game.won:
                break
        terminated = self.game.dead or self.game.won
        return self._obs(), 0.0, terminated, False, {}
