import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from settings import *
from sprites import Ball, Opponent, Player
from groups import AllSprites

class PongEnv(gym.Env):
    FPS = 30  # Add class constant
    DT = 1/FPS  # Also add timestep constant

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.cumulative_reward = 0
        self.prev_ball_pos = None
        
        # Action space (0: stay, 1: up, 2: down)
        self.action_space = spaces.Discrete(3)
        
        # Observation space (ball_x, ball_y, ball_dx, ball_dy, paddle_y)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0]),  # All values normalized
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        # Initialize Pygame components
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Initialize game components
        self.all_sprites = AllSprites()
        self.paddle_sprites = pygame.sprite.Group()
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cumulative_reward = 0
        
        # Reset score
        self.score = {'player': 0, 'opponent': 0}
        
        # Clear existing sprites
        self.all_sprites.empty()
        self.paddle_sprites.empty()
        
        # Create game objects
        self.ball = Ball(self.all_sprites, self.paddle_sprites, self.update_score)
        self.opponent_paddle = Opponent((self.all_sprites, self.paddle_sprites), self.ball)
        
        # Create and position the player paddle
        self.player_paddle = Player((self.all_sprites, self.paddle_sprites))
        self.player_paddle.rect.right = WINDOW_WIDTH - 20
        
        self.prev_ball_pos = np.array([self.ball.rect.centerx, self.ball.rect.centery])
        
        # Initial observation
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        # Get current ball position
        current_ball_pos = np.array([self.ball.rect.centerx, self.ball.rect.centery])
        
        # Calculate position differences (velocity)
        if self.prev_ball_pos is None:
            ball_dx, ball_dy = 0, 0
        else:
            ball_dx = (current_ball_pos[0] - self.prev_ball_pos[0]) / WINDOW_WIDTH
            ball_dy = (current_ball_pos[1] - self.prev_ball_pos[1]) / WINDOW_HEIGHT
        
        # Store current position for next frame
        self.prev_ball_pos = current_ball_pos

        obs = np.array([
            current_ball_pos[0] / WINDOW_WIDTH,
            current_ball_pos[1] / WINDOW_HEIGHT,
            ball_dx,  # Change in x position (normalized)
            ball_dy,  # Change in y position (normalized)
            self.player_paddle.rect.centery / WINDOW_HEIGHT
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        # Process action
        self.player_paddle.direction = {
            0: 0,     # Stay
            1: -1,    # Up
            2: 1      # Down
        }[action]

        # Update game state using class constant
        self.all_sprites.update(self.DT)

        # Calculate reward
        reward, terminated = self._calculate_reward()

        # Get new observation
        observation = self._get_observation()

        # Track cumulative reward (for logging only)
        self.cumulative_reward += reward

        return observation, reward, terminated, False, {}

    def _calculate_reward(self):
        reward = 0.0
        terminated = False

        # Check for direction change from positive to negative
        if self.ball.previous_direction.x > 0 and self.ball.direction.x < 0:
            reward = 0.1  # Reward for changing direction from positive to negative
            print("Reward for changing direction from positive to negative")

        # Game outcome rewards
        if self.score['player'] > 0:  # Ball passed opponent
            reward = 1.0
            terminated = True
        elif self.score['opponent'] > 0:  # Ball passed player
            reward = -1.0
            terminated = True

        # Reset scores if terminated
        if terminated:
            self.score['player'] = 0
            self.score['opponent'] = 0

        return reward, terminated

    def render(self):
        self.display_surface.fill(COLORS['bg'])
        self.all_sprites.draw()
        pygame.display.update()
        self.clock.tick(self.FPS)  # Use class constant

    def update_score(self, side):
        self.score[side] += 1

    def close(self):
        pygame.quit()
