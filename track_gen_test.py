import math
import random
import pygame
from scipy.interpolate import splprep, splev
import numpy as np

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

def draw_track(points, window_size = 800, draw_control_points = False):
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

    if draw_control_points:
        # Dibujar puntos sobre los vértices (debugging)
        for point in points:
            # Es necesario trabajar en número enteros y no en decimales (cada entero es un píxel)
            coord_x = int(point[0])
            coord_y = int(point[1])

            pygame.draw.circle(window, (255, 0, 0), (coord_x, coord_y), 6)

    pygame.event.pump()
    pygame.display.update()
    clock.tick(1)

def generate_smooth_track(x_center, y_center, num_points = 15, base_radius = 150):
    points_x = []
    points_y = []

    angle_dist = (2 * math.pi) / num_points

    for i in range(num_points):
        angle = i * angle_dist

        noise = random.uniform(-0.4, 0.4)
        current_radius = base_radius * (1 + noise)

        x = x_center + (math.cos(angle) * current_radius)
        y = y_center + (math.sin(angle) * current_radius)

        points_x.append(x)
        points_y.append(y)

    # Para que el circuito se cierre perfectamente tenemos que duplicar el primer punto al final de la lista
    points_x.append(points_x[0])
    points_y.append(points_y[0])

    # SUAVIZADO
    # s=0 significa que obligamos a la curva a pasar EXACTAMENTE por nuestros puntos
    # per=True le dice que es un circuito cerrado (periódico)
    tck, u = splprep([points_x, points_y], s = 0, per = True)

    # splev genera los nuevos puntos. 
    # np.linspace(0, 1, 100) crea 100 puntos a lo largo de esa curva.
    u_nuevo = np.linspace(0, 1, 100)
    smooth_points_x, smooth_points_y = splev(u_nuevo, tck)
        
    # Combinamos las X y las Y de nuevo en una lista de tuplas para Pygame
    track = list(zip(smooth_points_x, smooth_points_y))

    return track
    
if __name__ == "__main__":
    ventana_abierta = True
    clock = pygame.time.Clock()

    # draw_track(generate_base_track(400, 400))
    draw_track(generate_smooth_track(400, 400, 20))

    while ventana_abierta:
        # Capturamos todo lo que el usuario hace (teclado, ratón, etc.)
        for evento in pygame.event.get():
            # Si el evento es hacer clic en la 'X' de la ventana
            if evento.type == pygame.QUIT:
                ventana_abierta = False
        
        # Limitamos el bucle a 10 FPS para que tu procesador no se ponga al 100%
        clock.tick(10)

    

