"""Configuración común de pytest."""

import os

# Permite usar pygame sin ventana (CI, entornos de entrenamiento).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
