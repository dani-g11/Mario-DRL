# Mario DRL — Motor, entorno y aprendizaje por refuerzo

> **Por:** Daniel Giraldo, Diego Angarita, Esteban Alvarez, Ismael Garcia
> **Entrega final:** Introducción a la inteligencia artificial.

![Mario, un goomba y el HUD](docs/captures/entidades.png)

## 1. Introducción

Este proyecto implementa un juego 2D de plataformas inspirado en Super Mario Bros. y lo convierte en un entorno de aprendizaje por refuerzo. Un agente observa el estado del nivel, escoge acciones como avanzar, correr o saltar y recibe una recompensa según su progreso. El objetivo práctico es que aprenda a llegar a la bandera sin caer en fosos ni chocar con los goombas.

El aporte principal es separar el proyecto en capas claras. El motor define las reglas físicas y del juego; el nivel y el render convierten esas reglas en un escenario visible; el entorno expone una interfaz compatible con Gymnasium; y el entrenamiento permite evaluar si el agente aprende a completar el nivel. Esta separación permite probar el juego manualmente, validar la física y entrenar el agente sin mezclar responsabilidades.

## 2. Objetivo

### Objetivo general

Construir y evaluar un entorno de Mario en Python donde un agente de aprendizaje por refuerzo aprenda a avanzar por niveles de plataformas mediante interacción con un motor físico determinista.

### Objetivos específicos

- Implementar un motor con movimiento, salto, gravedad, colisiones, enemigos, cámara y reglas de victoria o derrota.
- Exponer el juego como un entorno Gymnasium con siete acciones, observaciones en rejilla o píxeles, recompensas y final de episodio.
- Medir el desempeño del agente mediante progreso, tasa de llegada a la bandera, causas de muerte y generalización a un segundo nivel.

## 3. Descripción de los datos

Este proyecto no utiliza un conjunto de datos externo tradicional. Los datos se generan durante la interacción del agente con el entorno.

| Elemento | Descripción |
|---|---|
| Fuente | Simulación procedimental del juego definida en el repositorio. |
| Formato del nivel | Matriz de tiles construida desde mapas ASCII. |
| Niveles | Dos niveles de 200 columnas y 15 filas. |
| Acciones | Siete acciones discretas: nada, derecha, saltar, correr y combinaciones. |
| Variables del estado | Posición, velocidad, contacto con el suelo, enemigos, progreso y estado del episodio. |
| Recompensa | Avance horizontal, penalización por paso, penalización por muerte y premio por bandera. |

### Preparación y generación

Los niveles se escriben con caracteres ASCII y se transforman en códigos internos de tile. El motor verifica que las filas tengan el mismo ancho, rellena el espacio superior con aire y valida que los obstáculos, fosos, checkpoints y enemigos estén en posiciones permitidas.

Durante cada episodio se reinician el tiempo, la posición de Mario, los enemigos, la cámara, la puntuación y los bloques modificables. Para las pruebas de robustez se pueden variar la velocidad de los goombas, el punto inicial y la repetición pegajosa de acciones.

### Limitaciones y restricciones

- Los niveles son diseñados por el equipo y no representan la variedad completa de los juegos de plataformas.
- Los resultados dependen de la semilla, el número de pasos de entrenamiento y la configuración de aleatoriedad.
- Los sprites son pixel-art del proyecto y no se incluyen assets comerciales externos.

## 4. Metodología

El trabajo se organiza en cinco etapas:

1. **Definición del mundo:** se fijan las dimensiones, velocidades, gravedad, tipos de tiles, reglas de puntuación y contrato entre módulos.
2. **Construcción del motor:** `MarioGame` actualiza la física de Mario y los goombas, resuelve las colisiones, mueve la cámara y determina muerte o victoria.
3. **Creación del entorno:** `MarioEnv` traduce las acciones del agente, mantiene cada acción durante cuatro frames, genera observaciones, calcula recompensas y marca el final del episodio.
4. **Entrenamiento:** se utiliza PPO con Stable-Baselines3, con entornos vectorizados, semillas, checkpoints, TensorBoard y soporte para observaciones estructuradas o imágenes.
5. **Evaluación:** se mide el progreso hacia la bandera, la tasa de victoria, las causas de muerte y la generalización en el nivel 2 y bajo condiciones aleatorias.

### Decisiones principales

- **Motor determinista:** con el mismo nivel, velocidad y secuencia de entradas se obtiene el mismo resultado. Esto facilita las pruebas y permite comparar experimentos.
- **Colisiones por ejes:** primero se corrige el movimiento horizontal y luego el vertical. Así Mario no atraviesa paredes, pisos ni techos.
- **Acciones discretas:** siete acciones reducen el espacio de decisiones y mantienen una interfaz sencilla para el agente.
- **Frame skip de cuatro frames:** cada decisión del agente se mantiene durante cuatro frames del motor, reduciendo el número de decisiones necesarias.
- **Dos observaciones:** `grid` ofrece una representación compacta del estado; `pixels` permite estudiar una entrada visual.
- **PPO:** se eligió la implementación disponible en Stable-Baselines3 para entrenar políticas con observaciones estructuradas o visuales y varios entornos en paralelo.
- **Métricas:** se usan tasa de llegada a la bandera, progreso medio y causas de muerte porque muestran tanto el éxito final como la calidad del avance.

### Diagrama de flujo metodológico

<img width="1433" height="598" alt="image" src="https://github.com/user-attachments/assets/3f510b63-3351-4c7a-8f64-829c6fb89859" />

> pipeline desde el diseño de reglas y niveles, pasando por el motor, el entorno Gymnasium y el entrenamiento PPO, hasta la evaluación y los resultados.

## 5. Diagrama de interacción UML

Durante un episodio, el agente solicita una acción al entorno. El entorno traduce esa acción a botones del motor. `MarioGame` actualiza la física, consulta el nivel, revisa colisiones y devuelve el nuevo estado. Después, el entorno calcula la recompensa y entrega al agente la siguiente observación. El proceso se repite hasta que Mario llega a la bandera, muere o el episodio se trunca.

Los componentes principales son:

- **Agente PPO:** elige la siguiente acción.
- **MarioEnv:** adapta el motor a la interfaz Gymnasium.
- **MarioGame:** ejecuta la simulación y las reglas.
- **Level:** contiene tiles, enemigos, inicio, bandera y checkpoints.
- **Render:** dibuja el estado cuando se solicita una vista.
- **Usuario:** observa el modo humano o el render de evaluación.

<img width="1491" height="858" alt="image" src="https://github.com/user-attachments/assets/ca9c84eb-cd7b-4371-9468-55984b100860" />

> secuencia temporal entre agente, entorno, motor, nivel, render y usuario durante un episodio.

## 6. Resultados

### Validación del motor y los niveles

El motor cuenta con pruebas de contrato, movimiento, salto, colisiones, cámara, enemigos, reglas y determinismo. Los niveles también se validan desde sus checkpoints con un bot heurístico y varias velocidades de goomba.

| Evaluación | Resultado documentado |
|---|---:|
| Nivel 1 con bot heurístico | 36/36 partidas ganadas |
| Nivel 2 con bot heurístico | 44/44 partidas ganadas |
| Nivel 1 con agente PPO sin aleatoriedad | 20/20 episodios, 100 % |
| Nivel 1 con acciones pegajosas | 20/20 episodios, 100 % |
| Nivel 1 con velocidad variable de goombas | 19/20 episodios, 95 % |
| Nivel 1 con ambas aleatoriedades | 17/20 episodios, 85 % |
| Nivel 2 para generalización | 16/20 episodios, 80 % |
| Línea base aleatoria | 0/30 episodios |
| Línea base heurística | 30/30 episodios |

Los resultados provienen de los informes versionados en `docs/validacion_nivel1.md`, `docs/validacion_nivel2.md` y `docs/informe_final.md`.

### Análisis

El motor y los niveles cumplen las restricciones físicas y son superables desde los checkpoints probados. El agente PPO aprende el nivel 1 con alta tasa de éxito cuando el entorno es determinista. Su rendimiento disminuye cuando se combinan acciones pegajosas y velocidades variables de goombas, y también baja en el nivel 2. Esto indica que el agente aprende una estrategia útil, pero todavía existe una diferencia entre memorizar un nivel y generalizar a una disposición nueva.

La línea base aleatoria confirma que completar el nivel no ocurre por azar. La línea base heurística muestra que la tarea es físicamente resoluble y sirve como referencia superior para comparar al agente aprendido.

### Limitaciones de los resultados

- Las cifras reportadas corresponden a experimentos concretos y no deben interpretarse como garantía para cualquier semilla.
- El nivel 2 reutiliza los mismos tipos de piezas y límites físicos, aunque cambia su orden y su geometría.
- El rendimiento visual puede requerir más entrenamiento que el de la observación `grid`.
- La recompensa favorece el avance horizontal, por lo que no es idéntica a la puntuación clásica del juego.

## 7. Reproducibilidad

### Requisitos

- Python 3.12 recomendado.
- Pygame 2.6.1.
- Dependencias listadas en `requirements.txt`.
- Windows, Linux o macOS con soporte para Python y los paquetes indicados.

### Instalación en Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux o macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Comandos mínimos

Jugar el nivel 1:

```powershell
python play.py
```

Jugar el nivel 2:

```powershell
python play.py --level 2
```

Mostrar un nivel en ASCII:

```powershell
python -m mario_game.level 1
```

Validar un nivel:

```powershell
python -m mario_game.validate --level 2 --goomba-speeds 0.3 0.5 1.0 1.5 --report docs/validacion_nivel2.md
```

Entrenar una prueba corta del pipeline:

```powershell
python train.py --cartpole --steps 100000
```

Entrenar y evaluar un agente PPO:

```powershell
python train.py --obs grid --steps 2000000 --name lvl1_grid_s0
python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20
```

Ejecutar las pruebas:

```powershell
python -m pytest
```

### Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `mario_game/constants.py` | Constantes físicas y códigos de tiles; Persona 1. |
| `mario_game/game.py` | Motor, física, colisiones, goombas, cámara y reglas; Persona 1. |
| `mario_game/level.py` | Niveles ASCII, checkpoints y representación del mapa; Persona 2. |
| `mario_game/sprites.py` y `render.py` | Sprites y dibujo del juego; Persona 2. |
| `play.py` | Modo humano con teclado; Persona 2. |
| `mario_env.py` | Entorno Gymnasium, acciones, observaciones y recompensa; Persona 3. |
| `train.py`, `evaluate.py` y `mario_training.py` | Entrenamiento, evaluación y métricas PPO; Persona 4. |
| `tests/` | Pruebas del motor, niveles, entorno, render y entrenamiento. |
| `docs/` | Informes, capturas y diagramas. |
| `.github/workflows/tests.yml` | Integración continua para pruebas y benchmark. |

## Responsabilidades del equipo

- **Persona 1:** motor del juego, constantes, física, colisiones, cámara y reglas.
- **Persona 2:** niveles, sprites, render, modo humano y validación visual.
- **Persona 3:** entorno Gymnasium, observaciones, acciones, recompensas y aleatoriedad.
- **Persona 4:** entrenamiento PPO, evaluación, experimentos, vídeos e informe final.

## Referencias internas

- [Contrato de interfaces](INTERFACES.md)
- [Informe final](docs/informe_final.md)
- [Validación del nivel 1](docs/validacion_nivel1.md)
- [Validación del nivel 2](docs/validacion_nivel2.md)
- [Diagrama de flujo metodológico](docs/diagrams/flujo_metodologico.svg)
- [Diagrama de secuencia](docs/diagrams/secuencia_interaccion.svg)
