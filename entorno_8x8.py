import gymnasium as gym
from gymnasium import spaces
import pygame

class Entorno_8x8(gym.Env):
    metada = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1
        }
    
    def __init__(self, render_mode = None):
        super().__init__()

        self.goal = [7, 7]
        self.position = [0, 0]

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Discrete(63)

        # GRÁFICOS
        self.render_mode = render_mode
        self.window_size = 800
        self.cell_size = self.window_size // 8 # mantenemos en 100 pixeles cada celda
        self.window = None
        self.clock = None
    
    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        self.position = [0, 0]
        info = {}

        if self.render_mode == "human":
            self.render()

        return self.position, info
    
    def step(self, action):
        # ACCIONES
        # 0 -> Arriba
        # 1 -> Derecha
        # 2 -> Abajo
        # 3 -> Izquierda

        match action:
            case 0:
                self.position[1] -= 1
            case 1:
                self.position[0] += 1
            case 2:
                self.position[1] += 1
            case 3:
                self.position[0] -= 1
        
        position_x = max(0, min(self.position[0], 7))
        position_y = max(0, min(self.position[1], 7))

        self.position = [position_x, position_y]

        terminated = bool(self.position[0] == self.goal[0] and self.position[1] == self.goal[1])

        # RECOMPENSAS
        if terminated:
            reward = 10.0
        else:
            reward = -1.0
        
        truncated = False # cambiar cuando funcione

        # Renderizado
        if self.render_mode == "human":
            self.render()

        return self.position, reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        
        if self.window is None:
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Entorno 8x8 Fijo")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        # Dibujar Cuadrícula
        for i in range(8):
            for j in range(8):
                pygame.draw.rect(
                    self.window,
                    (200, 200, 200),
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    2
                )

        # Dibujar meta
        pygame.draw.rect(
            self.window,
            (100, 255, 100),
            (self.goal[0] * self.cell_size, self.goal[1] * self.cell_size , self.cell_size, self.cell_size)
        )

        # Dibujar el agente
        centro_x = (self.position[0] * self.cell_size) + (self.cell_size // 2)
        centro_y = (self.position[1] * self.cell_size) + (self.cell_size // 2)
        pygame.draw.circle(self.window, (50, 150, 255), (centro_x, centro_y), 30)

        # Actualizamos la pantalla y forzamos los fps
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metada["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()

if __name__ == "__main__":
    # 1. Instanciamos nuestro entorno
    env = Entorno_8x8(render_mode = "human")
    
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
            action = env.action_space.sample()
            
            if action == 0:
                texto_accion = "Arriba(0)"
            elif action == 1:
                texto_accion = "Derecha(1)"
            elif action == 2:
                texto_accion = "Abajo(2)"
            elif action == 3:
                texto_accion = "Izquierda(3)"
            
            # 3. Aplicamos la acción al entorno
            nueva_obs, recompensa, terminado, truncado, info = env.step(action)
            
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