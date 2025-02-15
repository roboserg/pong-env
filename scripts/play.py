import pygame

from pong_env import PongEnv

env = PongEnv(render_mode="human")
observation, _ = env.reset()

running = True
while running:
    action = 0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        action = 1
    elif keys[pygame.K_DOWN]:
        action = 2
    
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if reward != 0:
        print(f"Step reward: {reward}")
        print(f"Cumulative reward: {env.cumulative_reward}")
    
    if terminated or truncated:
        observation, _ = env.reset()
        print("Resetting environment")

env.close()
