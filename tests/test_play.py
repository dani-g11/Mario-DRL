"""Tests de play.py (Persona 2, P2.8): el modo humano, sin abrir ninguna ventana.

La lógica (teclas, pausa, reinicio, carteles) está en `Session`, que solo recibe el
estado del teclado y una superficie. El bucle con ventana se prueba con el driver
de vídeo "dummy" que fija conftest.py.
"""

from collections import defaultdict

import pytest

pygame = pytest.importorskip("pygame")

import play  # noqa: E402
from mario_game import render  # noqa: E402
from mario_game.constants import SCREEN_H, SCREEN_W, TILE  # noqa: E402


def keys(*pressed):
    """Un 'teclado' con solo esas teclas pulsadas (como pygame.key.get_pressed())."""
    state = defaultdict(bool)
    for key in pressed:
        state[key] = True
    return state


def new_surface():
    return pygame.Surface((SCREEN_W, SCREEN_H))


def pixels(surface) -> bytes:
    return pygame.image.tobytes(surface, "RGB")


def key_event(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


# --- Teclado -------------------------------------------------------------------------

@pytest.mark.parametrize("pressed, expected", [
    ((), (False, False, False, False)),
    ((pygame.K_LEFT,), (True, False, False, False)),
    ((pygame.K_a,), (True, False, False, False)),
    ((pygame.K_RIGHT,), (False, True, False, False)),
    ((pygame.K_d,), (False, True, False, False)),
    ((pygame.K_SPACE,), (False, False, True, False)),
    ((pygame.K_UP,), (False, False, True, False)),
    ((pygame.K_w,), (False, False, True, False)),
    ((pygame.K_LSHIFT,), (False, False, False, True)),
    ((pygame.K_RSHIFT,), (False, False, False, True)),
    ((pygame.K_x,), (False, False, False, True)),
    ((pygame.K_RIGHT, pygame.K_SPACE, pygame.K_x), (False, True, True, True)),
])
def test_read_inputs_maps_keys_to_step_arguments(pressed, expected):
    assert play.read_inputs(keys(*pressed)) == expected


# --- Tamaño de la ventana -------------------------------------------------------------

def test_choose_scale_respects_the_requested_scale():
    assert play.choose_scale(2, (1920, 1080)) == 2
    assert play.choose_scale(5, (800, 600)) == 5      # lo pide el usuario: no se discute
    assert play.choose_scale(0, None) == 1            # nunca menos de 1


@pytest.mark.parametrize("desktop, expected", [
    ((1920, 1080), 3),     # pantalla grande: la escala por defecto
    ((3840, 2160), 3),     # nunca más de la escala por defecto
    ((1366, 768), 2),      # portátil: 3 × 240 = 720 no deja sitio a la barra de título
    ((1024, 600), 2),      # 2 × 240 = 480 + 120 de margen = 600: justo
    ((800, 480), 1),       # pantalla pequeña: ni 2 cabe
    ((200, 100), 1),       # absurdo: la mínima, no 0
    (None, 3),             # no se conoce el tamaño
    ((0, 0), 3),           # el driver dummy no tiene escritorio
])
def test_choose_scale_fits_the_desktop(desktop, expected):
    assert play.choose_scale(None, desktop) == expected


# --- La sesión ------------------------------------------------------------------------

def test_a_new_session_starts_at_the_level_start():
    session = play.Session()
    assert not session.paused and not session.finished
    assert session.game.frame == 0
    assert int(session.game.mario.x) // TILE == session.game.level.start_col


def test_update_advances_one_frame_with_the_pressed_keys():
    session = play.Session()
    x0 = session.game.mario.x
    session.update(keys(pygame.K_RIGHT))
    assert session.game.frame == 1
    assert session.game.mario.x > x0
    x1 = session.game.mario.x
    session.update(keys(pygame.K_LEFT))
    assert session.game.mario.x < x1 + 1.5            # empieza a frenar: no sigue acelerando


def test_pause_freezes_the_game():
    session = play.Session()
    session.handle_key(pygame.K_p)
    assert session.paused
    before = session.game.frame
    for _ in range(10):
        session.update(keys(pygame.K_RIGHT))
    assert session.game.frame == before
    session.handle_key(pygame.K_p)
    session.update(keys())
    assert not session.paused and session.game.frame == before + 1


def test_cannot_pause_once_the_episode_is_over():
    session = play.Session()
    session.game.dead = True
    session.handle_key(pygame.K_p)
    assert not session.paused


def test_r_restarts_and_keeps_the_chosen_start():
    session = play.Session()
    for _ in range(30):
        session.update(keys(pygame.K_RIGHT, pygame.K_x))
    session.handle_key(pygame.K_r)
    assert session.game.frame == 0
    assert int(session.game.mario.x) // TILE == session.game.level.start_col
    assert session.episode == 2


def test_restart_also_unpauses():
    session = play.Session()
    session.handle_key(pygame.K_p)
    session.handle_key(pygame.K_r)
    assert not session.paused


def test_enter_only_restarts_when_the_episode_is_over():
    session = play.Session()
    session.update(keys(pygame.K_RIGHT))
    session.handle_key(pygame.K_RETURN)
    assert session.game.frame == 1                    # sigue la misma partida
    session.game.dead = True
    session.handle_key(pygame.K_RETURN)
    assert session.game.frame == 0 and not session.finished


def test_digit_keys_jump_to_the_checkpoints():
    session = play.Session()
    checkpoints = session.game.level.checkpoints
    session.handle_key(pygame.K_3)
    assert int(session.game.mario.x) // TILE == checkpoints[2]
    session.handle_key(pygame.K_r)                    # R repite desde ese checkpoint
    assert int(session.game.mario.x) // TILE == checkpoints[2]
    session.handle_key(pygame.K_0)                    # 0 vuelve al inicio del nivel
    assert int(session.game.mario.x) // TILE == session.game.level.start_col


def test_a_checkpoint_that_does_not_exist_is_ignored():
    session = play.Session()
    session.update(keys(pygame.K_RIGHT))
    assert session.jump_to_checkpoint(len(session.game.level.checkpoints) + 1) is False
    assert session.game.frame == 1 and session.episode == 1


def test_every_checkpoint_of_the_level_can_be_started_from():
    session = play.Session()
    for number in range(1, len(session.game.level.checkpoints) + 1):
        assert session.jump_to_checkpoint(number)
        assert not session.finished


def test_start_col_is_validated():
    with pytest.raises(ValueError, match="foso"):
        play.Session(start_col=52)                    # un foso del nivel 1
    with pytest.raises(ValueError, match="fuera"):
        play.Session(start_col=9999)
    with pytest.raises(ValueError, match="nivel"):
        play.Session(level_number=99)


def test_start_col_is_used_by_the_session():
    session = play.Session(start_col=94)
    assert int(session.game.mario.x) // TILE == 94


def test_goomba_speed_is_passed_to_the_engine():
    assert play.Session(goomba_speed=1.0).game.goomba_speed == 1.0


# --- Fin de partida ---------------------------------------------------------------------

def test_the_summary_is_reported_once_per_episode():
    session = play.Session()
    assert session.update(keys()) is None
    session.game.dead, session.game.death_reason = True, "goomba"
    summary = session.update(keys())
    assert "Episodio 1" in summary and "goomba" in summary
    assert session.update(keys()) is None             # no se repite
    session.handle_key(pygame.K_r)
    session.game.won = True
    assert "Episodio 2" in session.update(keys()) and "victoria" in session.summary()


def test_banner_depends_on_the_state():
    session = play.Session()
    assert session.banner() is None
    session.paused = True
    assert session.banner()[0] == "PAUSA"
    session.paused = False
    session.game.dead, session.game.death_reason = True, "pit"
    title, lines, _ = session.banner()
    assert title == "GAME OVER" and "FOSO" in lines[0]
    session.game.death_reason = "time"
    assert "TIEMPO" in session.banner()[1][0]
    session.game.dead, session.game.won = False, True
    title, lines, _ = session.banner()
    assert title == "NIVEL COMPLETADO" and session.game.score == 0 and "000000" in lines[0]


def test_every_banner_text_uses_characters_of_the_font():
    session = play.Session()
    texts = set()
    for state in ("playing", "paused", "pit", "goomba", "time", "won"):
        session = play.Session()
        if state == "paused":
            session.paused = True
        elif state == "won":
            session.game.won = True
        elif state != "playing":
            session.game.dead, session.game.death_reason = True, state
        banner = session.banner()
        if banner:
            texts.update(banner[0] + "".join(banner[1]))
    assert texts <= set(render._GLYPHS), texts - set(render._GLYPHS)


def test_draw_shows_the_banner_on_top_of_the_game():
    session = play.Session()
    playing, over = new_surface(), new_surface()
    session.draw(playing)
    session.game.dead, session.game.death_reason = True, "pit"
    session.draw(over)
    assert pixels(playing) != pixels(over)
    plain = new_surface()
    session.game.render(plain)
    assert pixels(over) != pixels(plain)


def test_draw_matches_the_game_render_while_playing():
    session = play.Session()
    a, b = new_surface(), new_surface()
    session.draw(a)
    session.game.render(b)
    assert pixels(a) == pixels(b)


def test_no_hud_option_hides_the_hud():
    with_hud, without = play.Session(), play.Session(hud=False)
    a, b = new_surface(), new_surface()
    with_hud.draw(a)
    without.draw(b)
    assert pixels(a) != pixels(b)


# --- Eventos y bucle ----------------------------------------------------------------------

@pytest.mark.parametrize("event, keeps_running", [
    (pygame.event.Event(pygame.QUIT), False),
    (key_event(pygame.K_ESCAPE), False),
    (key_event(pygame.K_q), False),
    (key_event(pygame.K_p), True),
    (key_event(pygame.K_r), True),
    (pygame.event.Event(pygame.MOUSEMOTION, pos=(1, 1), rel=(0, 0), buttons=(0, 0, 0)), True),
])
def test_handle_event_says_whether_to_keep_running(event, keeps_running):
    assert play.handle_event(play.Session(), event) is keeps_running


def test_handle_event_passes_keys_to_the_session():
    session = play.Session()
    play.handle_event(session, key_event(pygame.K_p))
    assert session.paused


def test_main_runs_a_few_frames_without_a_window_manager(capsys):
    assert play.main(["--frames", "20", "--scale", "1"]) == 0
    assert capsys.readouterr().out.startswith("Controles:")


def test_main_with_start_col_and_options(capsys):
    assert play.main(["--frames", "5", "--scale", "1", "--start-col", "94",
                      "--no-hud", "--goomba-speed", "0.8"]) == 0
    assert "SHIFT" in capsys.readouterr().out         # imprime los controles


@pytest.mark.parametrize("args", [["--start-col", "52"], ["--start-col", "9999"],
                                  ["--scale", "0"], ["--level", "7"]])
def test_main_rejects_bad_arguments_with_a_message(args, capsys):
    with pytest.raises(SystemExit) as exc:
        play.main(args + ["--frames", "1"])
    assert exc.value.code == 2
    assert "error" in capsys.readouterr().err
