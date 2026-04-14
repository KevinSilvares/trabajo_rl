# Proyecto RL

## Conceptos

- `render_mode`: Modo de renderizado:
    - `human`: Imágenes y HUD con datos para humanos.
    - `rgb_array`: Imágenes rgb para entrenamiento más rápido.
    - `None`: Se incluye por defecto y no hace falta especificarlo. No renderizará nada. Entrena aún más rápido.
- `action_space`: Espacio de acción (acciones disponibles). Puede ser:
    - Discreto: Si las acciones van encadenadas una detrás de otra.
    - Continuo: Si las acciones pueden realizarse a la vez.
- `observation_space`: Espacio de observación. Lo que ve el modelo (pueden ser imágenes, datos, etc.).
- **Episodio**: Conjunto de acciones con un inicio y un resultado final (éxito/fallo).
- **Estado**: Situación actual. Dónde está el agente, qué variables tiene, etc. (ej: casilla 6, velocidad = 0.80).
- `reward`: La recompensa es por lo que se mueve el agente, intentando siempre maximizarla. Es un valor numérico. Esta recompensa se acumula por cada acción realizada. Puede ser positiva (si se ha hecho bien) o negativa (si se ha hecho mal). Pueden establecer distintas recompensas en función del hito/fracaso conseguido (ej: avanzar un tile = +1, completar el mapa = +100, retroceder tile = -0.5, agotar acciones disponibles = -100).

## Crear un entorno personalizado

Para crear un entorno personalizado de gymnasium se deben importar clases específicas y heredar de la clase `gym.Env`.

```python
import gymnasium as gym
from gymnasium import spaces
import pygame # sobre todo para visualización

class ArcadeMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
```

`render_fps`: Frames por segundo a los que se renderizará el entorno.

### Métodos necesarios

Al heredar de `gym.Env` se deben implementar unos métodos específicos.

#### `__init__()`

Constructor. En él se definen las reglas del entorno y el modo de renderizado.

```python
def __init__(self, render_mode=None):
        super().__init__() # llama al constructor de la clase padre
        
        # EL TABLERO: Una línea de 6 casillas (0, 1, 2, 3, 4, 5)
        self.meta = 5
        self.posicion = 0
        
        # ESPACIO DE ACCIÓN: Discreto(2) significa que la IA puede enviarnos 
        # un 0 (moverse izquierda) o un 1 (moverse derecha).
        self.action_space = spaces.Discrete(2)
        
        # ESPACIO DE OBSERVACIÓN: Discreto(6) significa que la IA solo recibe 
        # un número entero entre el 0 y el 5 diciéndole dónde está.
        self.observation_space = spaces.Discrete(6)
```

#### `reset()`

Devuelve el entorno a su estado original, finaliza el episodio actual e inicia uno nuevo. Esta función puede añadir aleatoriedad en el siguiente estado (como un cambio de mapa o de posición).

```python
def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Devolvemos el robot a la casilla 0
        self.posicion = 0 
        info = {} # se explica en el siguiente punto
        
        # reset siempre devuelve: (observacion, info)
        return self.posicion, info
```

#### `step()`

Da un paso adelante en la simulación en función del estado actual y de la acción que haya sido proporcionada. Esta función devuelve:

- `observation`: Estado actual. Dónde está el agente ahora.
- `reward`: Modificador para la `reward` acumulada por el agente hasta el momento.
- `terminated`: Booleano. Determina si el agente ha completado un episodio o no (ha triunfado o ha fracaso).
- `truncated`: Booleano. Determina si el agente ha agotado las acciones máximas o el tiempo máximo disponibles.
- `info`: Diccionario con información diagnóstica para medir el rendimiento del agente, debugging, variables del entorno…

```python
def step(self, accion):
        # 1. APLICAR LA ACCIÓN
        if accion == 0: # Izquierda
            self.posicion -= 1
        elif accion == 1: # Derecha
            self.posicion += 1
            
        # Evitamos que se salga del tablero por la izquierda o derecha
        self.posicion = max(0, min(self.posicion, 5))
        
        # 2. CALCULAR SI HA TERMINADO
        terminado = bool(self.posicion == self.meta)
        
        # 3. DAR RECOMPENSAS
        if terminado:
            recompensa = 10.0  # ¡Premio gordo por llegar!
        else:
            recompensa = -1.0  # Castigo por cada paso para que espabile y no tarde
            
        truncado = False # De momento no ponemos límite de tiempo
        
        # step siempre devuelve: (observacion, recompensa, terminado, truncado, info)
        return self.posicion, recompensa, terminado, truncado, {}
```

## Entorno de prueba con comentarios

Este entorno de prueba puede ejecutarse directamente sobre la misma clase. Representa una cuadrícula de 6x1 (6 casillas en una misma fila), un círculo (el agente) y un cuadrado verde (la meta).

No implementa ningún modelo de RL, por lo que las acciones son elegidas de forma aleatoria. Es sólo un entorno demostrativo.

```python
import gymnasium as gym
from gymnasium import spaces
import pygame

class PasilloEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode = None):
        super().__init__()

        # EL TABLERO: Una línea de 6 casillas (0, 1, 2, 3, 4, 5)
        self.meta = 5
        self.posicion = 0

        # ESPACIO DE ACCIÓN: Discreto(2) significa que la IA puede enviarnos
        # un 0 (moverse izquierda) o un 1 (moverse derecha).
        self.action_space = spaces.Discrete(2)

        # ESPACIO DE OBSERVACIÓN: Discreto(6) significa que la IA solo recibe
        # un número entero entre el 0 y el 5 diciéndole dónde está.
        self.observation_space = spaces.Discrete(6)

        # GRÁFICOS
        self.render_mode = render_mode
        self.window_size = 600
        self.cell_size = self.window_size // 6 # 100 píxeles por casilla
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Devolvemos el robot a la casilla 0
        self.posicion = 0 

        if self.render_mode == "human":
            self.render()

        # reset siempre devuelve: (observacion, info_extra)
        return self.posicion, {} 

    def step(self, accion):
        # 1. APLICAR LA ACCIÓN
        if accion == 0: # Izquierda
            self.posicion -= 1
        elif accion == 1: # Derecha
            self.posicion += 1

        # Evitamos que se salga del tablero por la izquierda o derecha
        self.posicion = max(0, min(self.posicion, 5))

        # 2. CALCULAR SI HA TERMINADO
        terminado = bool(self.posicion == self.meta)

        # 3. DAR RECOMPENSAS
        if terminado:
            recompensa = 10.0  # ¡Premio gordo por llegar!
        else:
            recompensa = -1.0  # Castigo por cada paso para que espabile y no tarde

        truncado = False # De momento no ponemos límite de tiempo

        # Para renderizarlos
        if self.render_mode == "human":
            self.render()

        # step siempre devuelve: (observacion, recompensa, terminado, truncado, info_extra)
        return self.posicion, recompensa, terminado, truncado, {}

    def render(self):
        # Si no se quiere renderizar hace early return
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.display.init()

            # Crea una ventana de 600 x 100 (que hemos configurado en el init)
            self.window = pygame.display.set_mode((self.window_size, self.cell_size))
            pygame.display.set_caption("Entorno test")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Dibujar el fondo
        self.window.fill((255, 255, 255))

        # Dibujar la cuadrícula
        for i in range(6):
            pygame.draw.rect(
                self.window,
                (200, 200, 200), # color (gris) del borde
                (i * self.cell_size, 0, self.cell_size, self.cell_size),
                2 # grosor de las líneas
            )

        # Dibujar la meta
        pygame.draw.rect(
            self.window,
            (100, 255, 100),
            (self.meta * self.cell_size, 0, self.cell_size, self.cell_size)
        )

        # Dibujar el agente
        centro_x = (self.posicion * self.cell_size) + (self.cell_size // 2)
        centro_y = self.cell_size // 2
        pygame.draw.circle(self.window, (50, 150, 255), (centro_x, centro_y), 30)

        # Actualizar pantalla y forzar los fps
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        # Limpia la memoria y borra la ventana
        if self.window is not None:
            pygame.quit()

# --- CÓDIGO DE PRUEBA ---

if __name__ == "__main__":
    # 1. Instanciamos nuestro entorno
    env = PasilloEnv(render_mode = "human")
    
    episodios = 3
    
    for episodio in range(episodios):
        print(f"\n=== INICIANDO EPISODIO {episodio + 1} ===")
        
        # Reseteamos el entorno y guardamos la posición inicial
        observacion, info = env.reset()
        print(f"Posición inicial: {observacion}")
        
        terminado = False
        puntuacion_total = 0
        paso = 1
        
        # Bucle que dura hasta que lleguemos a la meta
        while not terminado:
            # 2. Elegimos una acción aleatoria (0 o 1)
            accion = env.action_space.sample()
            
            texto_accion = "Izquierda (0)" if accion == 0 else "Derecha (1)"
            
            # 3. Aplicamos la acción al entorno
            nueva_obs, recompensa, terminado, truncado, info = env.step(accion)
            
            # 4. Imprimimos el resultado de este paso
            print(f"Paso {paso} | Acción: {texto_accion} -> Nueva posición: {nueva_obs} | Recompensa: {recompensa}")
            
            puntuacion_total += recompensa
            observacion = nueva_obs # Actualizamos la observación
            paso += 1
            
            # Medida de seguridad: Si el mono aleatorio es muy torpe y da 20 pasos 
            # sin llegar a la meta, cortamos el bucle para que no sea infinito.
            if paso > 20:
                print("¡El agente se ha perdido dando vueltas! Cortamos el episodio.")
                break
                
        print(f"--- Fin del Episodio {episodio + 1} | Puntuación Final: {puntuacion_total} ---")

```

## Generar pistas proceduralmente

### Conceptos

Para generar pistas proceduralmente es importante entender ciertos conceptos.

**Pista**: Una pista es un conjunto de puntos que se unen de principio a fin. Se crearán pistas en base a la técnica de rasterización polar. Esta técnica consiste traducir coordenadas polares (ángulo y radio) a cartesianas (x, y), lo que permite la creación de círculos con deformidades que serán usados como pistas.

**Círculo**: Un circulo, matemáticamente hablando, son 2 * Pi radianes (no 360º).

### Convertir de polar a cartesiana

Para convertir las coordenadas de polares a cartesianas se aplica la siguiente fórmula:

```python
x = centro_x + (math.cos(angulo) * radio_actual)
y = centro_y + (math.sin(angulo) * radio_actual)
```

Las funciones de coseno y seno toman como parámetros ángulos y devuelven siempre una proporción entre -1.0 y 1.0. Esta proporción indica cuanto se está moviendo un punto del eje correspondiente.

- cos: Se encarga del eje X (horizontal)
- sin: Se encarga del eje y (vertical)

La fórmula se explica:

1. Calculamos el cos y seno del ángulo correspondiente → obtenemos un número entre -1 y 1.
2. Lo multiplicamos por nuestro radio para hacer la distancia tan grande como queramos.
3. Sumamos la posición del centro en el eje correspondiente para que el punto se dibuje en el eje cartesiano.

### Ejemplo completo de generación de pista base

Con este código se genera una pista base (que es un círculo deformado de forma aleatoria en sus vértices) donde se marca en color verde el exterior (hierba), en gris el interior (asfalto) y los vértices se marcan en rojo.

```python
import math
import random
import pygame

def generate_base_track(x_center, y_center, num_points = 15, base_radius = 150):
    points = []
    
    # Matemáticamente un círculo no son 360º, son 2 * Pi radianes. Con esta línea se calcula el ángulo que separa cada uno de nuestros puntos en un círculo
    angle_dist = (2 * math.pi) / num_points

    for i in range(num_points):
        # Ángulo del punto iterado
        angle = i * angle_dist

        # Se aplica ruido para añadir variación en el radio del punto para no hacer un círculo perfecto.
        noise = random.uniform(-0.4, 0.4)
        current_radius = base_radius * (1 + noise)

        # Se convierten las coordenadas polares (angulo y radio) a cartesianas (x, y)
        x = x_center + (math.cos(angle) * current_radius)
        y = y_center + (math.sin(angle) * current_radius)

        points.append((x, y))
    
    return points

def draw_track(points, window_size = 800):
    pygame.init()
    pygame.display.init()

    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Pista Base Test")

    clock = pygame.time.Clock()

    # Pinta el fondo de verde oscuro
    window.fill((34, 139, 34))

    # Dibujar Asfalto
    if len(points) > 2:
        pygame.draw.polygon(
            window,
            (100, 100, 100),
            points
        )
    
    # Dibujar la línea de límite de pista
    pygame.draw.polygon(
        window,
        (255, 255, 255),
        points,
        3
    )

    # Dibujar puntos sobre los vértices (debugging)
    for point in points:
        # Es necesario trabajar en número enteros y no en decimales (cada entero es un píxel)
        coord_x = int(point[0])
        coord_y = int(point[1])

        pygame.draw.circle(window, (255, 0, 0), (coord_x, coord_y), 6)

    pygame.event.pump()
    pygame.display.update()
    clock.tick(1)

# Test
if __name__ == "__main__":
    ventana_abierta = True
    clock = pygame.time.Clock()

    draw_track(generate_base_track(400, 400))

    while ventana_abierta:
        # Capturamos todo lo que el usuario hace (teclado, ratón, etc.)
        for evento in pygame.event.get():
            # Si el evento es hacer clic en la 'X' de la ventana
            if evento.type == pygame.QUIT:
                ventana_abierta = False
        
        # Limitamos el bucle a 10 FPS para que tu procesador no se ponga al 100%
        clock.tick(10)
```

### Recursos

Documentación: [https://gymnasium.farama.org/environments/box2d/car_racing/](https://gymnasium.farama.org/environments/box2d/car_racing/) (Visto 10/04/2026)

Artículo resuelto: [https://www.findingtheta.com/es/blog/solving-gymnasiums-car-racing-with-reinforcement-learning](https://www.findingtheta.com/es/blog/solving-gymnasiums-car-racing-with-reinforcement-learning) (Visto 10/04/2026)

Generación de pistas de forma procedural: https://github.com/ChrisPHP/ProceduralRacetrack y [https://youtu.be/BTfghIWZFMw](https://youtu.be/BTfghIWZFMw) (Visto 12/04/2026)