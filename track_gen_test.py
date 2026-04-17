import math
import random
import pygame


def generate_base_track(x_center, y_center, num_points=15, base_radius=250):
    points = []

    # Matemáticamente un círculo no son 360º, son 2 * Pi radianes. Con esta línea se calcula el ángulo que separa cada uno de nuestros puntos en un círculo
    angle_dist = (2 * math.pi) / num_points

    for i in range(num_points):
        # Ángulo del punto iterado
        angle = i * angle_dist

        # Se aplica ruido para añadir variación en el radio del punto para no hacer un círculo perfecto.
        noise = random.uniform(-0.2, 0.2)
        current_radius = base_radius * (1 + noise)

        # Se convierten las coordenadas polares (angulo y radio) a cartesianas (x, y)
        x = x_center + (math.cos(angle) * current_radius)
        y = y_center + (math.sin(angle) * current_radius)

        points.append((x, y))

    return points


def draw_track(points, window_size=800):
    pygame.init()
    pygame.display.init()

    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Pista Base Test")

    clock = pygame.time.Clock()

    # Pinta el fondo de verde oscuro
    window.fill((34, 139, 34))

    # Dibujar Asfalto
    if len(points) > 2:
        pygame.draw.polygon(window, (200, 200, 200), points, 84)  # Borde blanco
        pygame.draw.polygon(window, (60, 60, 60), points, 80)  # Asfalto gris

        p1 = points[0]
        p2 = points[1]

        meta_rect = pygame.Surface((80, 20))
        meta_rect.fill((255, 255, 255))

        for i in range(0, 80, 20):
            pygame.draw.rect(meta_rect, (0, 0, 0), (i, 0, 10, 10))
            pygame.draw.rect(meta_rect, (0, 0, 0), (i + 10, 10, 10, 10))

        angle_meta = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        meta_rotada = pygame.transform.rotate(meta_rect, -angle_meta + 90)
        window.blit(meta_rotada, meta_rotada.get_rect(center=p1))

    # Dibujar la línea exterior de límite de pista
    pygame.draw.polygon(window, (255, 255, 255), points, 85)

    # Dibujar la línea interior de límite de pista
    pygame.draw.polygon(window, (255, 255, 255), points, 75)

    # Dibujar la línea de meta
    pygame.draw.polygon(window, (200, 200, 0), points, 2)

    # Dibujar puntos sobre los vértices (debugging)
    # for point in points:
    # Es necesario trabajar en número enteros y no en decimales (cada entero es un píxel)
    #   coord_x = int(point[0])
    #  coord_y = int(point[1])

    # pygame.draw.circle(window, (255, 0, 0), (coord_x, coord_y), 6)

    pygame.event.pump()
    pygame.display.update()
    clock.tick(1)


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
