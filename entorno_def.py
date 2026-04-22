import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np

from track_gen import generate_valid_track, generate_perfect_oval

class EntornoDef(gym.Env):
    FRICTION = 0.05
    
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, render_mode = None):
        super().__init__()
        self.max_inactivity_steps = 300
        self.inactivity_steps = 0

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
            shape = (self.obs_size, self.obs_size, 1), # 1 = 1 canales de color. Imágenes en blanco y negro.
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
    
    def _get_obs(self):
        # Crea el lienzo de la imagen
        obs_surface = pygame.Surface((self.window_size, self.window_size))

        # Pega el circuito (en RAM) sobre el lienzo
        obs_surface.blit(self.hitbox_surface, (0, 0))

        # "Dibuja" el coche como un cuadrado blanco puro para que destaque sobre el fondo
        car_rect = pygame.Surface((30, 15), pygame.SRCALPHA)
        car_rect.fill((255, 255, 255))

        rotated_car = pygame.transform.rotate(car_rect, self.rotation)
        rect = rotated_car.get_rect(center = (self.x, self.y))

        obs_surface.blit(rotated_car, rect.topleft)

        # Escala la imagen a 64x64
        small_surface = pygame.transform.scale(obs_surface, (self.obs_size, self.obs_size))

        # Extrae los píxeles RGB (3 canales)
        rgb_array = pygame.surfarray.array3d(small_surface)

        # Las redes neuronales prefieren Y,X,Color. La anterior línea devuelve X,Y,Color
        rgb_array = np.transpose(rgb_array, (1, 0, 2))

        # Convierte la imagen RGB a escala de grises calculando la media de los píxeles
        gray_array = np.mean(rgb_array, axis =2).astype(np.uint8)

        # Añade la dimensión (el canal de grises) a la imagen
        obs = np.expand_dims(gray_array, axis = -1)

        return obs 

    def reset(self, seed = None):
        super().reset(seed = seed)
        
        self.inactivity_steps += 1

        screen_center = self.window_size // 2       

        self.track_center, self.track_interior, self.track_exterior = generate_valid_track(screen_center, screen_center)
        # self.track_center, self.track_interior, self.track_exterior = generate_perfect_oval(self.window_size)

        # Coloca el pisto en la salida (El punto 0 de la pista)
        self.x = self.track_center[0][0]
        self.y = self.track_center[0][1]

        # Calcula la rotación del coche para que empiece mirando al punto 1 (igual que el cálculo de tangente de la pista)
        dx = self.track_center[1][0] - self.x
        dy = -(self.track_center[1][1] - self.y) # negativo porque las pantallas hacen crecer la Y hacia abajo
        self.rotation = math.degrees(math.atan2(dy, dx))

        self.speed = 0.0
        
        info = {}

        # CREACION DEL CIRCUITO EN MEMORIA
        self.hitbox_surface = pygame.Surface((self.window_size, self.window_size))

        # Crea el fondo de hierba
        self.grass_color = (34, 139, 34)
        self.hitbox_surface.fill(self.grass_color)

        # Pinta el aslfato
        num_points = len(self.track_center)
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

        # SISTEMA CHECKPOINTS
        # El coche aparece en el punto 0. Su objetivo es el punto 1 (y así en adelante)
        self.current_checkpoint = 1

        x_goal = self.track_center[self.current_checkpoint][0]
        y_goal = self.track_center[self.current_checkpoint][1]

        # Calcula la distancia a cada checkpoint para recompensar cuando se acerque
        self.previous_distance = math.hypot(self.x - x_goal, self.y - y_goal)

        self.num_checkpoints = len(self.track_center)

        self.inactivity_steps = 0
        return self._get_obs(), info
    
    def step(self, action):
        # pygame.event.pump() # Evita un bug que congelaba la ejecución sin renderizado
        self.inactivity_steps += 1

        steering_wheel = abs(action[0])
        pedal = action[1]

        engine_power = 1.0  # potencia del motor
        steering_sensibility = 10.0 # sensibilidad del giro. Responsividad del volante

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

        # Comprobación del estado con el circuito en memoria
        pixel_x = int(self.x)
        pixel_y = int(self.y)

        # Castigo por girar bruscamente
        reward = -0.05 - (steering_wheel * 0.1)

        terminated = False
        # Comprobación de si el coche se ha salido de la pantalla
        if pixel_x < 0 or pixel_x >= self.window_size or pixel_y < 0 or pixel_y >= self.window_size:
            terminated = True
            reward = -100.0
        else:
            # Comprueba el color que está pisando el coche
            track_position_color = self.hitbox_surface.get_at((pixel_x, pixel_y))
            color_rgb = (track_position_color[0], track_position_color[1], track_position_color[2])

            if color_rgb == self.grass_color:
                terminated = True
                reward = -50.0
            else:
                reward = -0.05 # aplicado porque el coche se estaba parando para no ser penalizado

                # COMRPOBAR CHECKPOINT
                # Saca el punto objetivo en x,y
                x_goal = self.track_center[self.current_checkpoint][0]
                y_goal = self.track_center[self.current_checkpoint][1]
                # Mide la distancia en línea recta (hipotenusa)
                current_distance = math.hypot(self.x - x_goal, self.y - y_goal)

                # Si la distancia es menor que la anterior se ha acercado (+) y le recompensamos por ello. Si no (-) le penalizamos
                best_distance = self.previous_distance - current_distance

                # Micro recompensa por avanzar
                reward += best_distance * 0.5

                if current_distance < 30.0: # por poner un margen. Realmente debería ser como la mitad del ancho que se establezca, pero de momento queda hardcodeado
                    reward += 10.0 # se premia el avance
                    
                    self.current_checkpoint += 1 # se actualiza el objetivo
                    self.inactivity_steps = 0
                    if self.current_checkpoint < self.num_checkpoints:
                        new_x_goal = self.track_center[self.current_checkpoint][0]
                        new_y_goal = self.track_center[self.current_checkpoint][1]
                        self.previous_distance = math.hypot(self.x - new_x_goal, self.y - new_y_goal)
                    else:
                        terminated = True
                        reward += 100.0 # mucha recompensa por completar el circuito
                        print("Vuelta completada.")
                
                if self.inactivity_steps >= self.max_inactivity_steps:
                    truncated = True
                    reward -= 20

        if self.render_mode == "human":
            self.render()

        truncated = False

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