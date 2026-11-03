"""Tests de mario_game/validate.py (Persona 2, P2.9): el bot y el informe de validación.

Dos partes:
- El validador funciona: el bot salta lo que debe, y detecta un nivel imposible.
- Cada nivel real (los de LEVEL_BUILDERS en test_level.py) se puede pasar desde cada
  checkpoint, también con goombas más lentos y más rápidos.
"""

import pytest

from mario_game import validate
from mario_game.constants import FPS, SCREEN_W, TILE
from mario_game.game import MarioGame
from mario_game.level import Level, tiles_from_ascii
from test_level import LEVEL_BUILDERS


def make_builder(art, **kwargs):
    """Una función que construye un nivel pequeño a partir de filas ASCII."""
    def build():
        defaults = dict(goomba_cols=[], start_col=2, flag_col=len(art[0]) - 6, checkpoints=[2])
        defaults.update(kwargs)
        return Level(tiles_from_ascii(art), **defaults)
    return build


GROUND = "#" * 40
FLAT = make_builder([GROUND, GROUND])
WIDE_PIT = make_builder(["#" * 12 + "." * 12 + "#" * 20] * 2)          # 12 de foso: no se salta
TALL_WALL = make_builder(["." * 20 + "H" + "." * 19] * 6 + [GROUND] * 2)   # 6 de alto: no se sube


# --- El bot ---------------------------------------------------------------------------

def test_the_bot_runs_right_on_flat_ground():
    game = MarioGame(FLAT())
    assert validate.Bot().act(game) == (False, True, False, True)


def test_the_bot_jumps_a_wall_in_front_of_it():
    game = MarioGame(make_builder(["....P....." + "." * 30, "....P....." + "." * 30,
                                   GROUND, GROUND])())
    assert validate.Bot().act(game)[2] is True


def test_the_bot_jumps_a_pit():
    game = MarioGame(make_builder(["#" * 5 + "..." + "#" * 32] * 2)())
    game.reset(start_col=2)
    for _ in range(200):
        _, _, jump, _ = validate.Bot().act(game)
        if jump:
            break
        game.step(False, True, False, True)
    assert jump, "el bot no saltó antes del foso"


def test_the_bot_releases_the_button_before_jumping_again():
    game = MarioGame(make_builder(["....P....." + "." * 30, "....P....." + "." * 30,
                                   GROUND, GROUND])())
    game.jump_held = True          # el salto del frame anterior sigue pulsado
    assert validate.Bot().act(game)[2] is False


def test_the_bot_jumps_a_goomba_that_is_close():
    game = MarioGame(make_builder([GROUND, GROUND], goomba_cols=[10])())
    goomba = game.goombas[0]
    goomba.active = True
    goomba.x = game.mario.x + game.mario.w + 30
    goomba.y = game.mario.y
    assert validate.Bot().act(game)[2] is True
    goomba.squash()                 # ya pisado: no es un peligro
    assert validate.Bot().act(game)[2] is False


# --- Una partida ----------------------------------------------------------------------

def test_the_bot_wins_a_flat_level():
    result = validate.play(FLAT, start_col=2)
    assert result.won and result.death_reason is None
    assert result.progress == 1.0 and result.outcome == "gana"
    assert result.seconds == result.frames / FPS
    assert result.end_col >= FLAT().flag_col - 1      # gana al entrar la caja de Mario en la columna


def test_the_bot_stomps_a_goomba_on_its_way():
    builder = make_builder([GROUND, GROUND], goomba_cols=[15])
    assert validate.play(builder, start_col=2).won


def test_a_pit_that_is_too_wide_is_reported_as_a_death():
    result = validate.play(WIDE_PIT, start_col=2)
    assert not result.won and result.death_reason == "pit"
    assert result.outcome == "muere (pit)"
    assert 12 <= result.end_col < 24                  # cae dentro del foso (columnas 12 a 23)


def test_a_wall_that_is_too_tall_is_reported_as_not_finished():
    result = validate.play(TALL_WALL, start_col=2, max_frames=600)
    assert not result.won and result.death_reason is None
    assert result.outcome == "sin terminar" and result.frames == 600
    assert result.end_col < 20


def test_play_is_deterministic():
    assert validate.play(FLAT, 2) == validate.play(FLAT, 2)


def test_play_does_not_share_the_level_between_games():
    calls = []

    def build():
        calls.append(1)
        return FLAT()
    validate.play(build, 2)
    validate.play(build, 2)
    assert len(calls) == 2


def test_play_from_an_invalid_column_raises():
    with pytest.raises(ValueError):
        validate.play(WIDE_PIT, start_col=13)          # sobre el foso


# --- Validar un nivel -----------------------------------------------------------------

def test_validate_level_runs_every_checkpoint_and_speed():
    builder = make_builder([GROUND, GROUND], checkpoints=[2, 10, 20])
    results = validate.validate_level(builder, goomba_speeds=(0.5, 1.0))
    assert [(r.start_col, r.goomba_speed) for r in results] == [
        (2, 0.5), (2, 1.0), (10, 0.5), (10, 1.0), (20, 0.5), (20, 1.0)]
    assert validate.all_passed(results)


def test_all_passed_is_false_for_failures_and_for_nothing():
    assert not validate.all_passed([])
    assert not validate.all_passed(validate.validate_level(WIDE_PIT))


# --- Informe --------------------------------------------------------------------------

def test_the_report_of_a_good_level():
    builder = make_builder([GROUND, GROUND], checkpoints=[2, 20])
    results = validate.validate_level(builder, (0.5,))
    report = validate.format_report("nivel de prueba", builder, results, (0.5,))
    assert report.startswith("# Validación de nivel de prueba")
    assert "| col. 2 |" in report and "| col. 20 |" in report
    assert "OK (" in report and "FALLA" not in report
    assert "2 de 2 partidas ganadas" in report
    assert "Todos los tramos son superables" in report


def test_the_report_of_a_bad_level_says_what_failed():
    results = validate.validate_level(WIDE_PIT, (0.5, 1.0))
    report = validate.format_report("malo", WIDE_PIT, results, (0.5, 1.0))
    assert report.count("FALLA: muere (pit)") == 2
    assert "0 de 2 partidas ganadas" in report
    assert "Hay tramos que el bot no supera" in report and "play.py" in report
    assert "Todos los tramos son superables" not in report


def test_the_report_numbers_the_stretches():
    # Un checkpoint de la columna 55 está en el tramo 1 (de 40 columnas cada uno).
    builder = make_builder([GROUND * 2, GROUND * 2], checkpoints=[2, 55], flag_col=70)
    report = validate.format_report("x", builder, validate.validate_level(builder), (0.5,))
    assert "| 0 | col. 2 |" in report and "| 1 | col. 55 |" in report


# --- Línea de comandos ------------------------------------------------------------------

def test_main_validates_level_1(capsys):
    assert validate.main(["--level", "1"]) == 0
    out = capsys.readouterr().out
    assert "Validación de nivel 1" in out and "9 de 9" in out


def test_main_writes_the_report_to_a_file(tmp_path, capsys):
    path = tmp_path / "informe.md"
    assert validate.main(["--report", str(path), "--goomba-speeds", "0.5", "1"]) == 0
    text = path.read_text(encoding="utf-8")
    assert "Goomba 0.5" in text and "Goomba 1" in text and "18 de 18" in text
    assert str(path) in capsys.readouterr().out


def test_main_returns_1_when_the_bot_fails(monkeypatch, capsys):
    monkeypatch.setattr(validate, "load_builder", lambda number: WIDE_PIT)
    assert validate.main([]) == 1
    assert "FALLA" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["--level", "99"], ["--goomba-speeds", "0"],
                                  ["--goomba-speeds", "0.5", "-2"]])
def test_main_rejects_bad_arguments(args, capsys):
    with pytest.raises(SystemExit) as exc:
        validate.main(args)
    assert exc.value.code == 2
    assert "error" in capsys.readouterr().err


def test_load_builder_finds_the_levels_by_number():
    assert validate.load_builder(1)().width > SCREEN_W // TILE
    with pytest.raises(ValueError, match="build_level_42"):
        validate.load_builder(42)


# --- Los niveles reales ----------------------------------------------------------------

@pytest.mark.parametrize("builder", LEVEL_BUILDERS, ids=lambda f: f.__name__)
def test_every_level_can_be_finished_from_every_checkpoint(builder):
    results = validate.validate_level(builder)
    failures = [f"checkpoint {r.start_col}: {r.outcome} en la columna {r.end_col}"
                for r in results if not r.won]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("builder", LEVEL_BUILDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("speed", [0.3, 1.0])
def test_levels_stay_passable_with_slower_and_faster_goombas(builder, speed):
    # P3.9 aleatoriza la velocidad de los goombas al entrenar: el nivel no puede
    # depender de un valor exacto.
    results = validate.validate_level(builder, goomba_speeds=(speed,))
    assert validate.all_passed(results), [(r.start_col, r.outcome) for r in results if not r.won]


class GoombaBlindBot(validate.Bot):
    """Un bot que no ve los goombas: sirve para comprobar que estorban."""
    GOOMBA_LOOKAHEAD = 0


@pytest.mark.parametrize("builder", LEVEL_BUILDERS, ids=lambda f: f.__name__)
def test_the_goombas_of_every_level_are_on_the_way(builder):
    # Si el bot pasa el nivel porque los goombas nunca le tocan, la validación no
    # prueba nada de ellos: sin saltarlos, el bot debe morir contra uno.
    result = validate.play(builder, builder().start_col, bot=GoombaBlindBot())
    assert not result.won and result.death_reason == "goomba"
