import pygame
from entorno_def import EntornoDef

def jugar_manual():
    env = EntornoDef(render_mode="human")
    obs, info = env.reset()

    env.render()

    print("--- MODO DEBUG MANUAL ---")
    print("Usa las FLECHAS del teclado para conducir.")
    print("Cierra la ventana para salir.")

    clock = pygame.time.Clock()
    corriendo = True

    while corriendo:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # Salir con la tecla ESC
        if keys[pygame.K_ESCAPE]:
            corriendo = False
            break

        # Controles mapeados al Action Space de tu IA
        volante = 0.0
        pedal = 0.0
        
        if keys[pygame.K_LEFT]:  volante = -1.0
        if keys[pygame.K_RIGHT]: volante =  1.0
        if keys[pygame.K_UP]:    pedal =  1.0
        if keys[pygame.K_DOWN]:  pedal = -1.0
        
        action = [volante, pedal]
        
        # Ejecutamos el paso
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"¡Te saliste de la pista! Recompensa: {reward}")
            obs, info = env.reset()
        elif truncated:
            print(f"¡Límite de inactividad alcanzado! Recompensa: {reward}")
            obs, info = env.reset()

        # Limitamos a 30 FPS para que sea jugable por un humano
        clock.tick(30)

    env.close()

if __name__ == "__main__":
    jugar_manual()