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
        self.prev_score = {'player': 0, 'opponent': 0}  # Add previous score tracking
        self.step_count = 0  # Initialize step counter
        
        # Action space (0: stay, 1: up, 2: down)
        self.action_space = spaces.Discrete(3)
        
        # Observation space (ball_x, ball_y, ball_dx, ball_dy, paddle_y)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -WINDOW_WIDTH, -WINDOW_HEIGHT, 0]),  # Raw values
            high=np.array([WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_HEIGHT]),
            dtype=np.float32
)

        # Initialize Pygame components
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Initialize fonts for different sizes
        self.score_font = pygame.font.Font(None, 72)  # Large font for score
        self.info_font = pygame.font.Font(None, 24)   # Smaller font for stats

        # Initialize game components
        self.all_sprites = AllSprites()
        self.paddle_sprites = pygame.sprite.Group()
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cumulative_reward = 0
        
        # Reset score
        self.prev_score = {'player': 0, 'opponent': 0}  # Reset previous score
        self.score = {'player': 0, 'opponent': 0}
        self.step_count = 0  # Reset step counter
        
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
            ball_dx = current_ball_pos[0] - self.prev_ball_pos[0]
            ball_dy = current_ball_pos[1] - self.prev_ball_pos[1]
        
        # Store current position for next frame
        self.prev_ball_pos = current_ball_pos

        obs = np.array([
            current_ball_pos[0],
            current_ball_pos[1],
            ball_dx,  # Change in x position
            ball_dy,  # Change in y position
            self.player_paddle.rect.centery
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        # Process action
        self.step_count += 1  # Increment step counter
        self.player_paddle.direction = {
            0: 0,     # Stay
            1: -1,    # Up
            2: 1      # Down
        }[action]

        # Update game state using class constant
        self.all_sprites.update(self.DT)

        # Calculate reward
        reward, terminated = self._calculate_reward()
        truncated = False  # Initialize truncated to False

        if self.step_count >= 10_000 and not terminated:
            truncated = True

        # Get new observation
        observation = self._get_observation()

        # Track cumulative reward (for logging only)
        self.cumulative_reward += reward

        return observation, reward, terminated, truncated, {}

    def _calculate_reward(self):
        reward = 0.0
        terminated = False

        # Check for direction change from positive to negative
        if self.ball.previous_direction.x > 0 and self.ball.direction.x < 0:
            reward = 0.01  # Reward for changing direction from positive to negative

        # Check if score changed this step
        if self.score['player'] > self.prev_score['player']:  # Player scored
            reward = 10.0
        elif self.score['opponent'] > self.prev_score['opponent']:  # Opponent scored
            reward = -1.0

        # Update previous score
        self.prev_score = self.score.copy()

        # Check if either player reached a score of 10
        if self.score['player'] >= 10 or self.score['opponent'] >= 10:
            terminated = True

        # Reset scores if terminated
        if terminated:
            self.score = {'player': 0, 'opponent': 0}
            self.prev_score = {'player': 0, 'opponent': 0}

        return reward, terminated

    def render(self):
        self.display_surface.fill(COLORS['bg'])
        self.all_sprites.draw()
        
        # Render scores with large font
        score_text = f"{self.score['opponent']}:{self.score['player']}"
        score_surface = self.score_font.render(score_text, True, COLORS['paddle'])
        score_rect = score_surface.get_rect(center=(WINDOW_WIDTH // 2, 50))
        self.display_surface.blit(score_surface, score_rect)
        
        # Render step count with smaller font
        step_count_text = f"Steps: {self.step_count}"
        step_count_surface = self.info_font.render(step_count_text, True, COLORS['paddle'])
        step_count_rect = step_count_surface.get_rect(topright=(WINDOW_WIDTH - 10, 10))
        self.display_surface.blit(step_count_surface, step_count_rect)
        
        # Render cumulative reward with smaller font
        reward_text = f"Reward: {self.cumulative_reward:.2f}"
        reward_surface = self.info_font.render(reward_text, True, COLORS['paddle'])
        reward_rect = reward_surface.get_rect(topright=(WINDOW_WIDTH - 10, 30))
        self.display_surface.blit(reward_surface, reward_rect)
        
        pygame.display.update()
        self.clock.tick(self.FPS)  # Use class constant

    def update_score(self, side):
        self.score[side] += 1

    def close(self):
        pygame.quit()
