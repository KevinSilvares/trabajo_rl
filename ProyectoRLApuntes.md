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

Constructor. En él se definen las reglas del entorno y el modo de renderizado.

#### `__init__()`

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

#### Ejemplo completo de generación de pista base

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

### Generación de pista suavizada

Partiendo de la base anterior, se aplica una interpolación entre los puntos que definen las curvas (vértices) de la pista. Esta técnica consiste en aplicar un spline o curvas matemáticas a la pista base (se recomienda buscar spline matemático). La idea es crear puntos entre los puntos ya existentes para suavizar las formas.

Para realizarlo se usa la librería `scipy` y dos funciones: `splprep` y `splev`.

```python
from scipy.interpolate import splprep, splev
```

El flujo es el siguiente:

#### `splprep`

Esta función crea el “plano” que utilizará la siguiente para poder dibujar los puntos:

```python
tck, u = splprep([puntos_x, puntos_y], s = 0, per = True)
```

**Argumentos:**

- Lista de puntos: Puntos (generados anteriormente) en x e y en una lista.
- `s`: smoothing. Controla cuanto de suave van a ser los puntos. Al contrario de lo que dice la lógica, realmente controla cuanto se puede desviar la curva que se va a crear de los puntos base. Al definirlo a 0 se le obliga a que la curva pase por todos los puntos definidos en la pista base.
- `per`: periódico. Indica si es una curva cerrada (`True`) o no (`False`).

**Devuelve:**

- `tck`: Una tupla que contiene todos los nodos, coeficientes y grado de la ecuación que deberá aplicarse después. En esencia, la configuración que necesitará `splev` después para poder aplicar la transformación. (No es necesario entender cada cosa que contiene este dato, sólo que tiene todas las matemáticas necesarias para aplicarse después por la otra función).
- `u`: Valores paramétricos originales (el porcentaje de recorrido original, no deberíamos usarlos).

El siguiente paso es espaciar los puntos. Cuantos más puntos haya disponibles más resolución tendrá la pista.

La letra `u` representa el porcentaje de recorrido realizado a lo largo de la curva.

```python
u_nuevo = np.linspace(0, 1, 100)
```

#### `splev`

Esta función aplica la transformación matemática sobre los valores paramétricos correspondientes (`u_nuevo`) con los parámetros almacenados en `tck`.

```python
puntos_suaves_x, puntos_suaves_y = splev(u_nuevo, tck)
```

Esta función evalúa cada punto de `u_nuevo` (el porcentaje recorrido) y calcula exactamente sus coordenadas cartesianas (x, y) en pantalla para ese punto concreto. Se devolverán tantas coordenadas x, y como puntos haya en `u_nuevo`.

#### Generar una pista suave con interior y exterior

No es suficiente con crear un mismo polígono y hacerlo más pequeño porque las coordenadas tendrían un offset (como una compensación) desde las coordenadas originales a las actuales y teniendo en cuenta la nueva escala.

La idea para generar una pista con interior e interior es partir de la base anterior para generar pistas suaves y usarla como puntos centrales para extruir a ambos lados.

1. Recorrer la lista de puntos que compone la pista.
2. Centrarnos en el punto A y el punto B. Formarán un vector (x, y) → (p1, p2).
3. Calcular la tangente del vector anterior. Formarán otro vector (dx, dy).
4. Normalizar la tangente para que mida 1 píxel siempre.
5. Obtener la normal de la tangente (básicamente girar la flecha a donde apunta la tangente para extruir a ambos lados).
6. Extruir (sumar la distancia) desde el punto central a cada lado con el ancho que viene como parámetro. Como partimos del centro, sumamos la mitad del ancho en una dirección y la otra mitad en la otra dirección.

```python
# resto del código anterior
for i in range(num_track_points):
        # Miramos el punto A (p1) y el punto B (p2)
        p1 = center_points[i]
        p2 = center_points[(i + 1) % num_track_points] # el % es para cerrar el círculo al final

        # Tangente
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Normalizar la tangente para que mida 1
        dist = math.hypot(dx, dy)
        if dist == 0: # puede llegar a dividir entre 0 y petaria
            dist = 1
        
        dx /= dist
        dy /= dist

        # Sacar la normal (girarlo 90º)
        nx = -dy
        ny = dx

        # Extruir
        # Borde interior
        half_width = width / 2

        interior.append((p1[0] + (nx * half_width), p1[1] + (ny * half_width)))
        exterior.append((p1[0] - (nx * half_width), p1[1] - (ny * half_width)))
```

Como aclaración: La idea es coger un punto central y fijarnos en el siguiente y obtener su tangente. La tangente es la dirección a la que apunta el vector formado por estos dos puntos. Al normalizarla obtenemos siempre una tangente de mismo valor; 1 píxel (de no hacerlo se obtendrían tangentes con muchos valores diferentes y no sería manejable). Luego giramos dicha flecha 90º y le sumamos la mitad del ancho a cada dirección.

#### Dibujar la pista

El proceso de dibujar la pista es muy similar a los anteriores pero con el matiz de que vamos a dibujar cada casilla o tile en lugar de una línea. Para ello se usa el método `polygon()` de `pygame` con los 4 puntos que forman la casilla.

Los límites de la pista se pueden dibujar como líneas ya que tenemos las listas de puntos interiores y exteriores.

```python
for i in range(num_points):
        next = (i + 1) % num_points # el % asegura que se cierre al final

        # Juntar los 4 puntos que forman un tile
        p1 = interior[i]
        p2 = exterior[i]
        p3 = exterior[next]
        p4 = interior[next]

        # Asfalto
        pygame.draw.polygon(
            window,
            (100, 100, 100),
            [p1, p2, p3, p4]
        )

        # Limites
        pygame.draw.lines(window, (255, 255, 255), True, interior, 3)
        pygame.draw.lines(window, (255, 255, 255), True, exterior, 3)
```

Es importante decir que este código no es perfecto y puede dar curvas raras o que se enroscan sobre sí mismas. Por simpleza, la forma de solucionarlo es aumentar el radio base en la generación de los puntos, el número de puntos (curvas) y/o la cantidad del ruido para cada punto. De base son:

- `num_points = 14`
- `base_radius = 250`
- `noise = (-0.2, 0.2)`

**Nota:** Si los puntos no se en el orden en el que lo hacen aparecerán como “pajaritas”. El pincel de pygame va esquina por esquina y rellena el interior del polígono, por lo que el orden a la hora de dibujar es importante. En este caso: PuntoA- Interior → PuntoA-Exterior → PuntoB-Exterior → PuntoB-Interior.

### Comprobación de posición en ejecución

Para comprobar donde se encuentra el coche durante la ejecución se utiliza el método `get_at(x, y)` de pygame. Este método toma como parámetros la posición en `x` e `y` y devuelve el píxel exacto. Con este píxel se puede comprobar su color:

- Si es gris el coche está en el asfalto y prosigue.
- Si es verde está en la hierba y se habrá salido, por lo que se se resetea la ejecución y se aplica una recompensan negativa.

Pero es importante destacar una cosa. Si el modo de renderizado no está en *human*, es decir, está en el modo `None`, que no renderiza nada y hace que el entrenamiento vaya miles de veces más rápido, no se creará ningún píxel y, por tanto, no habrá forma de calcular si el coche está tocando un píxel de asfalto o uno de hierba.

La forma de sortear esto es crear un mapa en memoria RAM, pero no renderizarlo en la función `render()`, sino originarlo en la función `reset()` y comprobarlo en cada invocación de `step()`. De este modo el circuito se utiliza, pero nunca se muestra al usuario. Todo queda en memoria.

#### En `reset()`

En esta función sólo se crea el circuto y se “pinta” de forma similar que en la función `render()`. La diferencia es que sólo lo almacenamos en la memoria pero no se lo mostramos al usuario.

```python
# creación del circuito y posicionamiento del coche por arriba

# CREACION DEL CIRCUITO EN MEMORIA
        self.hitbox_surface = pygame.Surface((self.window_size, self.window_size))

        # Crea el fondo de hierba
        self.grass_color = (34, 139, 34)
        self.hitbox_surface.fill(self.grass_color)

        # Pinta el aslfato
        num_points = len(self.track_interior)
        for i in range(num_points):
                next = (i + 1) % num_points
                p1 = self.track_interior[i]
                p2 = self.track_exterior[i]
                p3 = self.track_exterior[next]
                p4 = self.track_interior[next]

                # Asfalto
                pygame.draw.polygon(
                    self.hitbox_surface,
                    (100, 100, 100),
                    [p1, p2, p3, p4]
                )

        return np.array([self.x, self.y, self.rotation, self.speed]), info
```

Al crearse en una variable con `self`, esta va a ser accesible por toda la clase como un atributo más. Ahora existirá una variable `window`, que será donde se dibujen las cosas para el usuario, y una variable `hitbox_surface`, que se “dibujen” las cosas para el algoritmo. La diferencia está en que cuando el modo de render no esté en human no se creará `window`, pero siempre se creará `hitbox_surface` y permitirá al algoritmo entrenar sin que lo veamos.

#### En `step()`

La idea en esta función es comprobar el píxel que ocupa el coche sobre el circuito almacenado en `hitbox_surface`.

```python
# cálculo de posición, rotación, físicas aplicadas...

# Comprobación del estado con el circuito en memoria
        pixel_x = int(self.x)
        pixel_y = int(self.y)

        # Comprobación de si el coche se ha salido de la pantalla
        if pixel_x < 0 or pixel_x >= self.window_size or pixel_y < 0 or pixel_y >= self.window_size:
            terminated = True
            reward = -100.0
        else:
            # Comprueba el color que está pisando el coche
            track_position_color = self.hitbox_surface.get_at((pixel_x, pixel_y)) # Devuelve el píxel
            color_rgb = (track_position_color[0], track_position_color[1], track_position_color[2]) # Saca los canales R, G, B del píxel

            if color_rgb == self.grass_color:
                terminated = True
                reward = -50.0
            else:
                reward = 0.1

        if self.render_mode == "human":
            self.render()

        truncated = False

        obs = np.array([self.x, self.y, self.rotation, self.speed])
        return obs, reward, terminated, truncated, {}
```

En este ejemplo se da una recompensa de 0.1 por no tocar hierba, lo que en el futuro dará problemas porque el algoritmo aprenderá a estar únicamente quieto o a apenas moverse. Es válido en este ejemplo, pero lleva al siguiente punto.

### Comprobar que el coche avanza

Para esto se crea un sistema de checkpoints o puntos de control que el coche debe ir cruzando.

Se parte de la base de que en la creación de las pistas se obtiene un `track_center`. Esta es la lista de puntos centrales sobre los que luego se crean los bordes interior y exterior explicado en el punto anterior. Esta lista no deja de ser una lista de puntos del circuito, por lo que es perfectamente válida para comprobar cuantos puntos de control recorre el coche.

#### En `reset()`

En esta función simplemente se arranca el sistema.

```python
# resto de configuración de parámetros y ciruito en memoria

# SISTEMA CHECKPOINTS
# El coche aparece en el punto 0. Su objetivo es el punto 1 (y así en adelante)
self.current_checkpoint = 1
self.num_checkpoints = len(self.track_center)
        
return np.array([self.x, self.y, self.rotation, self.speed]), info        
```

#### En `step()`

En esta función se calcula la posición del coche respecto a su punto objetivo, un nuevo objetivo y se recompensa su avance.

```python
# en pseudocodigo la parte que ya se ha visto arriba

if coche_fuera_de_pantalla:
	terminated = True
  reward = -100.0
else:
	track_position_color = self.hitbox_surface.get_at((pixel_x, pixel_y))
	color_rgb = (track_position_color[0], track_position_color[1], track_position_color[2])
	
	if color_rgb == hierba:
		terminated = True
		reawrd = -50.0
	else:
		# COMPROBAR CHECKPOINT
		# Saca el punto objetivo en x,y
		x_goal = self.track_center[self.current_checkpoint][0]
    y_goal = self.track_center[self.current_checkpoint][1]

    # Mide la distancia en línea recta (hipotenusa)
    dist = math.hypot(self.x - x_goal, self.y - y_goal)
    if dist < 30.0: # por poner un margen. Realmente debería ser como la mitad del ancho que se establezca, pero de momento queda hardcodeado
	    reward += 10.0 # se premia el avance
                    
      self.current_checkpoint += 1 # se actualiza el objetivo
	      if self.current_checkpoint >= self.num_checkpoints:
	        terminated = True
          reward += 100.0 # mucha recompensa por completar el circuito
          print("Vuelta completada.")
```

El proceso de la comprobación del checkpoint es:

1. Se obtiene la coordenada x,y del punto objetivo (`x_goal`, `y_goal`)
2. Se mide la distancia desde el punto actual al punto objetivo. Como hemos visto antes, esta distancia es una línea recta y se calcula con la hipotenusa de estos puntos.
3. Si la distancia calculada es menor a el margen de error (en este caso 30.0) se entiende que el coche ha avanzado y se le premia con una pequeña recompensa. 
    
    **Nota:** Este margen existe porque el coche no va a ir por el centro la mayor parte del tiempo. Por lo que le damos un área para poder marcar el checkpoint aunque no pase exactamente por el punto.
    
4. Se actualiza el punto objetivo al siguiente.
5. Se comprueba que el punto objetivo (ya actualizado) no sea mayor o igual al número de checkpoints.
    1. Si no lo es el bucle se repite.
    2. Si lo es el coche ha terminado la vuelta. Se finaliza el episodio y se le da una recompensa muy grande por completar el objetivo final.

### Recursos

Documentación: [https://gymnasium.farama.org/environments/box2d/car_racing/](https://gymnasium.farama.org/environments/box2d/car_racing/) (Visto 10/04/2026)

Artículo resuelto: [https://www.findingtheta.com/es/blog/solving-gymnasiums-car-racing-with-reinforcement-learning](https://www.findingtheta.com/es/blog/solving-gymnasiums-car-racing-with-reinforcement-learning) (Visto 10/04/2026)

Generación de pistas de forma procedural: https://github.com/ChrisPHP/ProceduralRacetrack y [https://youtu.be/BTfghIWZFMw](https://youtu.be/BTfghIWZFMw) (Visto 12/04/2026)

[https://www.pygame.org/docs/](https://www.pygame.org/docs/) (Visto durante todo el proyecto)