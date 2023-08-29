from locale import normalize
import gymnasium as gym
import numpy as np
import math
import pygame
from physics import PhysicsObject, Jet

class JetFighterEnv(gym.Env):
    """Jet Fighter Environment that follows gym interface."""

    metadata = {'name' : 'jet_fighter'}

    def __init__(self):
        super().__init__()
        self.render_mode = "human"
        self.dim = 1000
        self.grid_size = (self.dim, self.dim)

        self.action_space = gym.spaces.MultiDiscrete([2, 3])
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.screen = None
        self.seed_value = None

    def reset(self, seed=None, options=None):
        self.seed_value = seed
        np.random.seed(seed)

        self.jet = Jet(self.grid_size)
        self.jet.randomize_position()
        self.jet.randomize_velocity()
        self.jet.randomize_heading()
        self.jet.randomize_angular_velocity()
 
        self.target = PhysicsObject(self.grid_size)
        self.target.max_linear_velocity = 10
        self.target.randomize_position()
        self.target.randomize_velocity()

        return self.get_obs(), {}

    def normalize_position(self, pos):
        return self.normalize_cyclic(pos, self.dim)

    def normalize_rads(self, rads):
        if np.isscalar(rads):
            return np.array([np.cos(rads), np.sin(rads)])
        else:
            return np.concatenate([np.cos(rads), np.sin(rads)])
    
    def normalize_cyclic(self, x, high):
        rads = math.tau * x / high
        return self.normalize_rads(rads)
    
    def normalize_scalar(self, x, low, high):
        return (2 * (x - low) / (high-low)) - 1
    
    def get_obs(self):
        return np.concatenate(
            [
                self.normalize_position(self.jet.relative_position_of(self.target)), # 4
                self.normalize_position(self.jet.wrapped_distance(self.target)), # 2
                self.normalize_rads(self.jet.relative_bearing_of(self.target)) # 2
            ]
        )

    def step(self, action):
        if action[0] == 1:
            self.jet.thrust()
        
        if action[1] == 1:
            self.jet.spin_ccw()
        elif action[1] == 2:
            self.jet.spin_cw()

        self.jet.update()
        self.target.update()

        reward, done = self.check_target()
        truncated = done

        observation = self.get_obs()
        return observation, reward, done, truncated, {}

    def check_target(self):
        distance = self.jet.wrapped_distance(self.target)
        if distance < 20:
            r, d = 1, True
        else:
            r, d = -1 - (distance / self.dim), False
        return r, d

    def render(self):
        if self.screen is None:
            print("initializing screen")
            self.window_size = (self.dim, self.dim)
            self.screen = pygame.display.set_mode(self.window_size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)

        self.screen.fill(BLACK)
        self.jet.render(self.screen, WHITE)
        self.target.render(self.screen, RED)

        pygame.display.flip()

    def close(self):
        # pygame.quit()
        return
