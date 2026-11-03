# Mario DRL

Juego de plataformas 2D estilo Super Mario Bros en Python + Pygame, pensado para
que un agente de aprendizaje por refuerzo (PPO, Stable-Baselines3) aprenda a
completar el nivel. El contrato entre módulos está en [INTERFACES.md](INTERFACES.md).

![Mario, un goomba y el HUD](docs/captures/entidades.png)

## Instalación (Python 3.12)

El equipo usa **Python 3.12** (pygame 2.6.1 y PyTorch no tienen soporte fiable
en 3.13/3.14) y el CI también. En Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # en cmd: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

En Linux/macOS: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

Comprueba que `(.venv)` aparece en la línea de comandos antes de usar `pip`: si
no, se instala en el Python global.

Si no tienes la 3.12, el juego y los tests también funcionan con Python 3.13
(`py -3.13 -m venv .venv`), pero el entrenamiento se hace con la 3.12.

## Jugar

```powershell
python play.py                  # el nivel 1, con el teclado
python play.py --level 2        # el nivel 2
python play.py --start-col 94   # empezar en una columna concreta (para probar un tramo)
python play.py --help           # todas las opciones (--scale, --goomba-speed, --no-hud)
```

| Tecla | Acción |
|---|---|
| Flechas o A / D | Moverse |
| Espacio, flecha arriba o W | Saltar (mantenerlo da un salto más alto) |
| Shift o X | Correr |
| P | Pausa |
| R (o Enter al terminar) | Reiniciar |
| 1-9 / 0 | Reiniciar desde el checkpoint n / desde el inicio del nivel |
| Esc o Q | Salir |

## Niveles

Hay dos niveles de 200 columnas (3200 px), con la bandera en la columna 186:

- **Nivel 1** (`build_level_1()`): el de entrenamiento. Fosos, tuberías, una pirámide y una escalera.
- **Nivel 2** (`build_level_2()`): mide la generalización. Mismas piezas y mismos límites, pero
  en otro orden y con otro ritmo, y con una meseta que el nivel 1 no tiene.

Los dos cumplen los límites del contrato (foso de 3 tiles como máximo, obstáculos de 4 como
máximo) y las reglas de diseño que comprueba `tests/test_level.py`. Para añadir un nivel
basta con una función `build_level_N()` en `mario_game/level.py` y añadirla a
`LEVEL_BUILDERS` en ese test.

```powershell
python -m mario_game.level 2          # el nivel 2 en texto, por tramos de 40 columnas
python scripts/export_captures.py     # capturas PNG de los sprites y de cada tramo, en docs/captures/
```

### Validar un nivel

Un bot intenta llegar a la bandera desde cada checkpoint con el motor real, con
varias velocidades de goomba. Si lo consigue en todos los casos, el nivel es superable.

```powershell
python -m mario_game.validate --level 2 --goomba-speeds 0.3 0.5 1.0 1.5 --report docs/validacion_nivel2.md
```

Termina con código 0 si el bot gana en todos los casos y 1 si falla alguno. Es determinista:
si cambian los niveles o la física del motor, hay que volver a generar los informes de
[`docs/validacion_nivel1.md`](docs/validacion_nivel1.md) y
[`docs/validacion_nivel2.md`](docs/validacion_nivel2.md).

## Tests y rendimiento

```powershell
python -m pytest              # suite completa (sin ventana: SDL_VIDEODRIVER=dummy)
python -m pytest -v tests/test_contract.py   # solo los límites del contrato
python scripts/benchmark_engine.py           # frames/s del motor sin gráficos
```

El CI de GitHub (`.github/workflows/tests.yml`) ejecuta los tests y el benchmark en
cada `push` y en cada PR, con Python 3.12.

## Estructura y responsables

| Ruta | Contenido | Responsable |
|---|---|---|
| `mario_game/constants.py` | Constantes compartidas | Persona 1 |
| `mario_game/game.py` | Motor: física, colisiones, goombas, reglas | Persona 1 |
| `mario_game/level.py` | Clase `Level`, `build_level_1()` y `build_level_2()` | Persona 2 |
| `mario_game/sprites.py`, `mario_game/render.py` | Sprites y dibujo del juego (`render.draw`) | Persona 2 |
| `mario_game/validate.py` | Bot e informe de validación de niveles | Persona 2 |
| `play.py` | Modo humano con teclado | Persona 2 |
| `mario_env.py` | Entorno Gymnasium | Persona 3 |
| `train.py`, `evaluate.py` | Entrenamiento y evaluación | Persona 4 |
| `tests/` | Tests del motor, los niveles, el render, `play.py` y el validador (`helpers.py` trae un nivel ASCII de prueba) | Personas 1 y 2 |
| `scripts/benchmark_engine.py` | Medición de velocidad del motor | Persona 1 |
| `scripts/export_captures.py` | Capturas PNG de sprites y niveles | Persona 2 |
| `docs/` | Capturas (`captures/`) e informes de validación de los niveles | Persona 2 |
| `.github/workflows/tests.yml` | CI: tests y benchmark en cada push y PR | Persona 1 |

## Uso mínimo del motor

```python
from mario_game.game import MarioGame
from mario_game.level import build_level_1

game = MarioGame(build_level_1())    # o build_level_2(). Un Level por juego: el motor modifica los bloques ?
game.reset()                         # o reset(start_col=c) desde un checkpoint (level.checkpoints)
while not (game.dead or game.won):
    game.step(left=False, right=True, jump=False, run=True)   # 1 frame
print(game.progress, game.death_reason)
```

Para dibujar un frame (sin ventana, por ejemplo para la observación `pixels`):

```python
import pygame
from mario_game.constants import SCREEN_H, SCREEN_W

surface = pygame.Surface((SCREEN_W, SCREEN_H))
game.render(surface)                 # no modifica el juego; el escalado lo hace quien llama
```
