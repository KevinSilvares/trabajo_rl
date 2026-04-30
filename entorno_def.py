import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np

from track_gen import generate_valid_track, generate_perfect_oval

class EntornoDef(gym.Env):
    # PHYSICS
    FRICTION = 0.05

    # CAR SETTINGS
    MAX_SPEED = 10.0
    ENGINE_POWER = 1.0
    BRAKE_POWER = 4.0
    STEERING_SENSIBILITY = 10.0
    
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, render_mode = None, track_type = "oval"):
        super().__init__()
        self.max_inactivity_steps = 100
        self.inactivity_steps = 0

        self.track_type = track_type

        # ACCIONES: Volante y Pedales
        # Ambos controlen van desde el -1.0 al 1.0
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0]),
            high = np.array([1.0, 1.0]),
            dtype = np.float32
        )

        # Espacio de observación (vectores de estado)
        self.obs_size = 7

        # pos_x, pos_y, velocidad, sin_orientación, cos_orientación, pos_x_objetivo, pos_y_objetivo
        self.observation_space = spaces.Box(
            low = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]),
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype = np.float32
        )

        # ESTADO COCHE
        self.x = 300.0
        self.y = 300.0
        self.rotation = 90.0 # apunta arriba
        self.speed = 0.0

        # GRÁFICOS
        self.render_mode = render_mode
        self.window_size = 800
        self.cell_size = self.window_size // 8 # mantenemos en 100 pixeles cada celda # REVISAR.
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        # Normaliza los valores para que la IA aprenda mejor (entre 0 y 1 o -1 y 1)
        norm_x = self.x / self.window_size
        norm_y = self.y / self.window_size
        norm_speed = self.speed / self.MAX_SPEED
        
        # Pasa el ángulo a radianes para calcular seno y coseno
        rad = math.radians(self.rotation)
        sin_rotation = math.sin(rad)   
        cos_rotation = math.cos(rad)   

        # Calcula distancia en x,y al siguiente checkpoint
        if hasattr(self, "current_checkpoint") and self.current_checkpoint < len(self.track_center):
            x_goal = self.track_center[self.current_checkpoint][0]
            y_goal = self.track_center[self.current_checkpoint][1]
        else:
            x_goal = self.x
            y_goal = self.y

        norm_x_goal = (x_goal - self.x) / self.window_size
        norm_y_goal = (y_goal - self.y) / self.window_size


        obs =  np.array([
            norm_x, 
            norm_y, 
            norm_speed, 
            sin_rotation, 
            cos_rotation,
            norm_x_goal,
            norm_y_goal
        ], dtype=np.float32)
    
        return obs

    def reset(self, seed = None):
        super().reset(seed = seed)

        screen_center = self.window_size // 2

        if self.track_type == "oval":
            self.track_center, self.track_interior, self.track_exterior = generate_perfect_oval(self.window_size)
        elif self.track_type == "procedural":
            self.track_center, self.track_interior, self.track_exterior = generate_valid_track(screen_center, screen_center)
        else:
            print(f"No se reconoce '{self.track_type}' como una pista válida. Prueba: 'oval' ó 'procedural'.")

        # Coloca el pisto en la salida (El punto 0 de la pista)
        self.x = self.track_center[0][0]
        self.y = self.track_center[0][1]

        # Calcula la rotación del coche para que empiece mirando al punto 1 (igual que el cálculo de tangente de la pista)
        dx = self.track_center[1][0] - self.x
        dy = -(self.track_center[1][1] - self.y) # negativo porque las pantallas hacen crecer la Y hacia abajo
        self.rotation = math.degrees(math.atan2(dy, dx))

        self.speed = 0.0
        
        # SISTEMA CHECKPOINTS
        # El coche aparece en el punto 0. Su objetivo es el punto 1 (y así en adelante)
        self.current_checkpoint = 1
        self.num_checkpoints = len(self.track_center)
        self.inactivity_steps = 0

        info = {}
        return self._get_obs(), info
    
    def step(self, action):
        self.inactivity_steps += 1

        steering_wheel = action[0]
        pedal = action[1]

        if pedal > 0:
            # VELOCIDAD
            self.speed += pedal * self.ENGINE_POWER
        else:
            self.speed += pedal * self.BRAKE_POWER
        
        # Limitación de velocidad y elimina la marcha atrás
        self.speed = np.clip(self.speed, 0.0, 10.0)

        # GIRO
        if abs(self.speed) > 0.01:
            rotation = steering_wheel * self.STEERING_SENSIBILITY
            self.rotation -= rotation

        # FRICCION
        self.speed *= (1.0 - self.FRICTION)

        # FÍSICAS
        radians = math.radians(self.rotation) # para calcular las físicas necesitamos pasar los grados a radianes
        self.x += self.speed * math.cos(radians)
        # Se resta porque en pantallas el punto izquierdo alto es y = 0 
        self.y -= self.speed * math.sin(radians)
        
        # RECOMPENSAS
        # Recompensa por ir rápido.
        reward = (self.speed / 10.0) * 0.1

        # Castigo por girar bruscamente
        reward -= abs(steering_wheel) * 0.01

        terminated = False
        truncated = False

        # Comprobación de si el coche se ha salido de la pantalla
        if self.x < 0 or self.x >= self.window_size or self.y < 0 or self.y >= self.window_size:
            terminated = True
            reward = -1.0
        else:
            # COMPROBAR MATEMÁTICAMENTE QUE EL COCHE ESTÁ EN EL ASFALTO
            # Calcula la distancia al punto central más cercano
            min_dist_to_center = min(math.hypot(self.x - px, self.y - py) for px, py in self.track_center)
            track_radius = 30.0 # CAMBIAR EL MAGIC NUMBER DESPUÉS

            if min_dist_to_center > track_radius:
                terminated = True
                reward = -1.0
            else:
                # # COMRPOBAR CHECKPOINT
                # # Saca el punto objetivo en x,y
                x_goal = self.track_center[self.current_checkpoint][0]
                y_goal = self.track_center[self.current_checkpoint][1]

                # # Mide la distancia en línea recta (hipotenusa)
                current_distance = math.hypot(self.x - x_goal, self.y - y_goal)

                if current_distance < 30.0: # por poner un margen. Realmente debería ser como la mitad del ancho que se establezca, pero de momento queda hardcodeado
                    reward += 1.0 + (self.current_checkpoint * 0.5) # se premia el avance. Cuanto más lejos llega más puntos obtiene
                    
                    self.current_checkpoint += 1 # se actualiza el objetivo
                    self.inactivity_steps = 0

                    if self.current_checkpoint >= self.num_checkpoints:
                        terminated = True
                        reward += 10.0 # mucha recompensa por completar el circuito
                        print("Vuelta completada.")
                
                # PENALIZA SER EXTREMADAMENTE LENTO/QUEDARSE QUIETO
                if self.inactivity_steps >= self.max_inactivity_steps:
                    truncated = True
                    reward -= 5.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}
    
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

        # # VISUALIZACIÓN CHECKPOINT
            # if hasattr(self, "current_checkpoint") and self.current_checkpoint < len(self.track_center):
            #     target_x = self.track_center[self.current_checkpoint][0]
            #     target_y = self.track_center[self.current_checkpoint][1]
                
            #     # Dibuja un punto azul en el checkpoint actual
            #     pygame.draw.circle(self.window, (0, 191, 255), (int(target_x), int(target_y)), 8)
                
            #     # Dibuja una línea desde el coche hasta el checkpoint
            #     pygame.draw.line(self.window, (0, 191, 255), (int(self.x), int(self.y)), (int(target_x), int(target_y)), 2)

        # # DEBUG RADIO MATEMÁTICO
        #     track_radius = 30.0
        #     for px, py in self.track_center:
        #         pygame.draw.circle(self.window, (255, 0, 0), (int(px), int(py)), int(track_radius), 1)

        # Coche
        car_surface = pygame.Surface((30, 15), pygame.SRCALPHA)
        car_surface.fill((255, 0, 0))

        # Dibuja unos faros amarillos para saber cuál es el morro
        pygame.draw.rect(car_surface, (255, 255, 0), (26, 2, 4, 3))
        pygame.draw.rect(car_surface, (255, 255, 0), (26, 10, 4, 3))

        rotated_car = pygame.transform.rotate(car_surface, self.rotation)

        rotated_rect = rotated_car.get_rect(center = (self.x, self.y))

        self.window.blit(rotated_car, rotated_rect.topleft)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])