# Mario DRL

Juego de plataformas 2D estilo Super Mario Bros en Python + Pygame, pensado para
que un agente de aprendizaje por refuerzo (PPO, Stable-Baselines3) aprenda a
completar el nivel. El contrato entre módulos está en [INTERFACES.md](INTERFACES.md).

## Instalación (Python 3.12)

El equipo usa **Python 3.12** (pygame 2.6.1 y PyTorch no tienen soporte fiable
en 3.13/3.14). En Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # en cmd: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

En Linux/macOS: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

Comprueba que `(.venv)` aparece en la línea de comandos antes de usar `pip`: si
no, se instala en el Python global.

## Tests y rendimiento

```powershell
python -m pytest              # suite completa (sin ventana: SDL_VIDEODRIVER=dummy)
python -m pytest -v tests/test_contract.py   # solo los límites del contrato
python scripts/benchmark_engine.py           # frames/s del motor sin gráficos
```

## Estructura y responsables

| Ruta | Contenido | Responsable |
|---|---|---|
| `mario_game/constants.py` | Constantes compartidas | Persona 1 |
| `mario_game/game.py` | Motor: física, colisiones, goombas, reglas | Persona 1 |
| `mario_game/level.py`, `sprites.py`, `render.py`, `play.py` | Nivel, gráficos, modo humano | Persona 2 |
| `mario_env.py` | Entorno Gymnasium | Persona 3 |
| `train.py`, `evaluate.py` | Entrenamiento y evaluación | Persona 4 |
| `tests/` | Tests del motor (`helpers.py` trae un nivel ASCII de prueba) | Persona 1 |
| `scripts/benchmark_engine.py` | Medición de velocidad del motor | Persona 1 |

## Uso mínimo del motor

```python
from mario_game.game import MarioGame

game = MarioGame(level)              # level: objeto Level (INTERFACES.md §3)
game.reset()                         # o reset(start_col=c) desde un checkpoint
while not (game.dead or game.won):
    game.step(left=False, right=True, jump=False, run=True)   # 1 frame
print(game.progress, game.death_reason)
```
