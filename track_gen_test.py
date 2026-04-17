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

def draw_complete_track(center_points, interior, exterior, window_size = 800):
    pygame.init()
    pygame.display.init()

    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Pista Base Test")

    clock = pygame.time.Clock()

    # Pinta el fondo de verde oscuro
    window.fill((34, 139, 34))

    num_points = len(interior)

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

def generate_complete_track(x_center, y_center, num_points = 14, base_radius = 250, width = 60):
    # IGUAL QUE ANTES
    points_x = []
    points_y = []
    angle_dist = (2 * math.pi) / num_points

    for i in range(num_points):
        angle = i * angle_dist

        noise = random.uniform(-0.2, 0.2)
        current_radius = base_radius * (1 + noise)

        x = x_center + (math.cos(angle) * current_radius)
        y = y_center + (math.sin(angle) * current_radius)

        points_x.append(x)
        points_y.append(y)
    
    points_x.append(points_x[0])
    points_y.append(points_y[0])

    tck, u = splprep([points_x, points_y], s = 0, per = True)

    u_nuevo = np.linspace(0, 1, 100)
    smooth_points_x, smooth_points_y = splev(u_nuevo, tck)

    # Los puntos suavizados, lo que generate_smooth_track era la pista van a ser los puntos centrales para calcular el ancho de la pista
    center_points = list(zip(smooth_points_x, smooth_points_y))

    # EXTRUSION
    interior = []
    exterior = []

    num_track_points = len(center_points)

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

    return center_points, interior, exterior


if __name__ == "__main__":
    ventana_abierta = True
    clock = pygame.time.Clock()

    # draw_track(generate_base_track(400, 400))
    # draw_track(generate_smooth_track(400, 400, 20))
    center_points, interior, exterior = generate_complete_track(400, 400)
    draw_complete_track(center_points, interior, exterior)

    while ventana_abierta:
        # Capturamos todo lo que el usuario hace (teclado, ratón, etc.)
        for evento in pygame.event.get():
            # Si el evento es hacer clic en la 'X' de la ventana
            if evento.type == pygame.QUIT:
                ventana_abierta = False
        
        # Limitamos el bucle a 10 FPS para que tu procesador no se ponga al 100%
        clock.tick(10)

    

