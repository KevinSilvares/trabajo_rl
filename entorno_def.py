import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np

from track_gen import generate_complete_track

class EntornoDef(gym.Env):
    FRICTION = 0.05
    
    metadata = {
        "render_modes": ["human"],
        "render_fps": 10
    }

    def __init__(self, render_mode = None):
        super().__init__()

        # ACCIONES: Volante y Pedales
        # Ambos controlen van desde el -1.0 al 1.0
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0]),
            high = np.array([1.0, 1.0]),
            dtype = np.float32
        )

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

        screen_center = self.window_size // 2       

        self.track_center, self.track_interior, self.track_exterior = generate_complete_track(screen_center, screen_center)

        # Coloca el pisto en la salida (El punto 0 de la pista)
        self.x = self.track_center[0][0]
        self.y = self.track_center[0][1]

        # Calcula la rotación del coche para que empiece mirando al punto 1 (igual que el cálculo de tangente de la pista)
        dx = self.track_center[1][0] - self.x
        dy = -(self.track_center[1][1] - self.y) # negativo porque las pantallas hacen crecer la Y hacia abajo
        self.rotation = math.degrees(math.atan2(dy, dx))

        self.speed = 0.0
        
        info = {}

        return np.array([self.x, self.y, self.rotation, self.speed]), info
    
    def step(self, action):
        steering_wheel = action[0]
        pedal = action[1]

        engine_power = 2.0  # potencia del motor
        steering_sensibility = 5.0 # sensibilidad del giro. Responsividad del volante

        # VELOCIDAD
        self.speed += pedal * engine_power

        # Limitación de velocidad
        self.speed = np.clip(self.speed, -5.0, 15.0)

        # GIRO
        if abs(self.speed) > 0.1:
            rotation = steering_wheel * steering_sensibility

            # En caso de ir marcha atrás el giro se invierte
            if self.speed < 0:
                rotation = -rotation

            self.rotation -= rotation

        # FRICCION
        self.speed *= (1.0 - self.FRICTION)

        # FÍSICAS
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
            pygame.display.set_caption("Entorno Definitivo Test")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Rellena el fondo con hierba
        self.window.fill((34, 139, 34))

        # Dibuja la pista
        if hasattr(self, "track_interior") and len(self.track_interior) > 0:
            num_points = len(self.track_interior)

            for i in range(num_points):
                next = (i + 1) % num_points
                p1 = self.track_interior[i]
                p2 = self.track_exterior[i]
                p3 = self.track_exterior[next]
                p4 = self.track_interior[next]

                # Asfalto
                pygame.draw.polygon(
                    self.window,
                    (100, 100, 100),
                    [p1, p2, p3, p4]
                )
                
            # Limites
            pygame.draw.lines(self.window, (255, 255, 255), True, self.track_interior, 3)
            pygame.draw.lines(self.window, (255, 255, 255), True, self.track_exterior, 3)

        # Coche
        car_surface = pygame.Surface((30, 15), pygame.SRCALPHA)
        car_surface.fill((255, 0, 0))

        # Dibujamos unos faros amarillos para saber cuál es el morro
        pygame.draw.rect(car_surface, (255, 255, 0), (26, 2, 4, 3))
        pygame.draw.rect(car_surface, (255, 255, 0), (26, 10, 4, 3))

        rotated_car = pygame.transform.rotate(car_surface, self.rotation)

        rotated_rect = rotated_car.get_rect(center = (self.x, self.y))

        self.window.blit(rotated_car, rotated_rect.topleft)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
                            
if __name__ == "__main__":
    # 1. Instanciamos nuestro entorno
    env = EntornoDef(render_mode = "human")
    
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
            
            print(action)
            
            # 3. Aplicamos la acción al entorno
            nueva_obs, recompensa, terminado, truncado, info = env.step(action)
            
            # 4. Imprimimos el resultado de este paso
            print(f"Paso {paso} | Volante: {action[0]}, Pedal: {action[1]} -> Nueva posición: {nueva_obs} | Recompensa: {recompensa}")
            
            puntuacion_total += recompensa
            observacion = nueva_obs # Actualizamos la observación
            paso += 1
            
            # Medida de seguridad: Si el mono aleatorio es muy torpe y da 20 pasos 
            # sin llegar a la meta, cortamos el bucle para que no sea infinito.
            if paso > 200:
                print("¡El agente se ha perdido dando vueltas! Cortamos el episodio.")
                break
                
        print(f"--- Fin del Episodio {episodio + 1} | Puntuación Final: {puntuacion_total} ---")