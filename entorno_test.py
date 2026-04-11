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
