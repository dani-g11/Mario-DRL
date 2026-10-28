"""MarioGame.render(): delegación en mario_game/render.py (Persona 2).

render.py todavía no existe, así que se prueba con un módulo falso que
cumple la interfaz draw(surface, game, hud). También se comprueba que la
simulación no necesita pygame (INTERFACES.md §5).
"""

import subprocess
import sys
import types

import pytest

from helpers import make_level
from mario_game.constants import SCREEN_H, SCREEN_W
from mario_game.game import MarioGame


def new_game():
    return MarioGame(make_level(["#" * 40] * 2, goomba_cols=[10], start_col=1))


@pytest.fixture
def fake_renderer(monkeypatch):
    calls = []
    module = types.ModuleType("mario_game.render")
    module.draw = lambda surface, game, hud: calls.append((surface, game, hud))
    monkeypatch.setitem(sys.modules, "mario_game.render", module)
    return calls


def test_render_delegates_with_hud_flag(fake_renderer):
    game = new_game()
    surface = object()
    game.render(surface)
    game.render(surface, hud=False)
    assert fake_renderer == [(surface, game, True), (surface, game, False)]


def test_render_does_not_change_game_state(fake_renderer):
    game = new_game()
    for _ in range(30):
        game.step(False, True, False, True)
    before = (game.mario.x, game.mario.vx, game.cam_x, game.frame, game.time_left)
    game.render(object())
    assert (game.mario.x, game.mario.vx, game.cam_x, game.frame, game.time_left) == before


def test_render_works_with_a_headless_pygame_surface(fake_renderer):
    pygame = pytest.importorskip("pygame")
    surface = pygame.Surface((SCREEN_W, SCREEN_H))     # sin ventana ni display
    new_game().render(surface)
    assert fake_renderer[0][0] is surface


def test_missing_render_module_gives_clear_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "mario_game.render", raising=False)
    try:
        import mario_game.render  # type: ignore  # noqa: F401
        pytest.skip("render.py ya existe (Persona 2)")
    except ModuleNotFoundError:
        pass
    with pytest.raises(ImportError, match="render.py"):
        new_game().render(object())


def test_simulation_does_not_import_pygame():
    # En un proceso limpio: crear el juego y simular no debe cargar pygame.
    code = (
        "import sys; sys.path[:0] = ['.', 'tests']\n"
        "from helpers import make_level\n"
        "from mario_game.game import MarioGame\n"
        "g = MarioGame(make_level(['#' * 40] * 2, goomba_cols=[10]))\n"
        "for _ in range(300): g.step(False, True, True, True)\n"
        "print('pygame' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False"
