
import math
import random
import numpy as np
from scipy.interpolate import splprep, splev

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
