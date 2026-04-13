import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np

class EntornoLibre(gym.Env):
    FRICTION = 0.05
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10
    }

    def __init__(self, render_mode = None):
        super().__init__()

        # ACCIONES (0: Nada, 1: Acelerar, 2: Frenar, 3: Izquierda, 4: Derecha)
        self.action_space = spaces.Discrete(5)

        # Espacio de observación (una imagen de 64x64 píxeles)
        self.obs_size = 64

        self.observation_space = spaces.Box(
            low = 0, # negro
            high = 255, # blanco
            shape = (self.obs_size, self.obs_size, 3), # 3 = 3 canales de color. Esto puede cambiar a 1 para que se entrene con imágenes en blanco y negro.
            dtype = np.uint8 # uint8 en lugar de int para mejorar la eficiencia de RAM. El rango es de 0 a 255.
        )

        # ESTADO COCHE
        self.x = 300.0
        self.y = 300.0
        self.rotation = 90.0 # apunta arriba
        self.speed = 0.0


        # GRÁFICOS
        self.render_mode = render_mode
        self.window_size = 800
        self.cell_size = self.window_size // 8 # mantenemos en 100 pixeles cada celda
        self.window = None
        self.clock = None

    def reset(self, seed = None):
        super().reset(seed = seed)

        self.x = 300.0
        self.y = 300.0
        self.rotation = 90.0 # apunta arriba
        self.speed = 0.0

        info = {}

        return np.array([self.x, self.y, self.rotation, self.speed]), info
    
    def step(self, action):
        if action == 1: self.speed += 1.5
        if action == 2: self.speed -= 1.5
        if action == 3: self.rotation += 1.5
        if action == 4: self.rotation -= 1.5

        self.speed *= (1.0 - self.FRICTION)

        radians = math.radians(self.rotation) # para calcular las físicas necesitamos pasar los grados a radianes

        self.x += self.speed * math.cos(radians)

        # Se resta porque en pantallas el punto izquierdo alto es y = 0 
        self.y -= self.speed * math.sin(radians)

        terminated = False
        truncated = False
        reward = 0.0

        if self.render_mode == "human":
            self.render()

        obs = np.array([self.x, self.y, self.rotation, self.speed])
        return obs, reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        
        if self.window is None:
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Entorno Libre")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((50, 50, 50))

        car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
        car_surface.fill((255, 0, 0))

        # Dibujamos unos faros amarillos para saber cuál es el morro
        pygame.draw.rect(car_surface, (255, 255, 0), (35, 2, 5, 5))
        pygame.draw.rect(car_surface, (255, 255, 0), (35, 13, 5, 5))

        rotated_car = pygame.transform.rotate(car_surface, self.rotation)

        rotated_rect = rotated_car.get_rect(center = (self.x, self.y))

        self.window.blit(rotated_car, rotated_rect.topleft)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])


if __name__ == "__main__":
    # 1. Instanciamos nuestro entorno
    env = EntornoLibre(render_mode = "human")
    
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
                texto_accion = "Nada(0)"
            elif action == 1:
                texto_accion = "Acelerar(1)"
            elif action == 2:
                texto_accion = "Frenar(2)"
            elif action == 3:
                texto_accion = "Izquierda(3)"
            elif action == 4:
                texto_accion = "Derecha(4)"
            
            # 3. Aplicamos la acción al entorno
            nueva_obs, recompensa, terminado, truncado, info = env.step(action)
            
            # 4. Imprimimos el resultado de este paso
            print(f"Paso {paso} | Acción: {texto_accion} -> Nueva posición: {nueva_obs} | Recompensa: {recompensa}")
            
            puntuacion_total += recompensa
            observacion = nueva_obs # Actualizamos la observación
            paso += 1
            
            # Medida de seguridad: Si el mono aleatorio es muy torpe y da 20 pasos 
            # sin llegar a la meta, cortamos el bucle para que no sea infinito.
            if paso > 200:
                print("¡El agente se ha perdido dando vueltas! Cortamos el episodio.")
                break
                
        print(f"--- Fin del Episodio {episodio + 1} | Puntuación Final: {puntuacion_total} ---")
