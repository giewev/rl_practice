from locale import normalize
import gymnasium as gym
import numpy as np
import math
import pygame

def deg_sin(x):
    return math.sin(math.radians(x))


def deg_cos(x):
    return math.cos(math.radians(x))

class PhysicsObject:
    def __init__(self, env_dimensions):
        self.dim = env_dimensions
        self.position = np.zeros(2)
        self.linear_velocity = np.zeros(2)
        self.rotation = 0
        self.rotational_velocity = 0


class Jet(PhysicsObject):
    def __init__(self, env_dimensions):
        super(Jet, env_dimensions)
        self.position = np.random.uniform(0, self.dim, 2)
        self.rotation = np.random.uniform(0, 360)

class JetFighterEnv(gym.Env):
    """Jet Fighter Environment that follows gym interface."""

    metadata = {}

    def __init__(self):
        super().__init__()
        self.render_mode = "human"
        self.dim = 1000
        self.grid_size = (self.dim, self.dim)
        self.linear_thrust = 1
        self.rotational_thrust = 1
        self.max_rotation_velocity = 10
        self.max_linear_velocity = 30
        self.rotation_damping = 0.99
        self.linear_damping = 0.99
        self.max_target_velocity = 5

        # self.action_space = gym.spaces.MultiDiscrete(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2, 3])
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        self.screen = None
        self.seed_value = None

    def reset(self, seed=None, options=None):
        self.seed_value = seed
        np.random.seed(seed)

        self.jet_position = np.random.uniform(0, self.dim, 2)
        self.jet_angle = np.random.uniform(0, 360)
        self.target_position = np.random.uniform(0, self.dim, 2)
        target_angle = np.random.uniform(0, math.tau)
        self.target_velocity = self.max_target_velocity * np.array([np.cos(target_angle), np.sin(target_angle)])

        self.linear_velocity = np.zeros(2)
        self.rotational_velocity = 0
        self.shot_time = 0
        self.step_count = 0
        self.just_fired = False

        return self.get_obs(), {}

    def normalize_position(self, pos):
        return self.normalize_cyclic(pos, self.dim)

    def normalize_degrees(self, angle):
        return self.normalize_cyclic(angle, 360)
    
    def normalize_cyclic(self, x, high):
        rads = math.tau * x / high
        if np.isscalar(x):
            return np.array([np.cos(rads), np.sin(rads)])
        else:
            return np.concatenate([np.cos(rads), np.sin(rads)])
    
    def normalize_scalar(self, x, low, high):
        return (2 * (x - low) / (high-low)) - 1
    
    def wrapped_distance(self, pos_1, pos_2):
        delta = np.abs(pos_1 - pos_2)
        wrapped_delta = np.minimum(delta, self.dim - delta)
        return np.linalg.norm(wrapped_delta)

    def get_obs(self):
        return np.concatenate(
            [
                self.normalize_position(self.jet_position), # 4
                self.normalize_degrees(self.jet_angle), # 2
                self.normalize_position(self.target_position), # 4
                # self.normalize_position(self.linear_velocity), # 4
                # self.normalize_degrees(self.rotational_velocity), # 2
                # self.normalize_position(self.target_velocity), #4
            ]
        )

    def step(self, action):
        lin = 0
        rot = 0
        self.just_fired = False
        if action[0] == 1:
            lin = self.linear_thrust
        
        if action[1] == 1:
            rot = self.rotational_thrust
        elif action[1] == 2:
            rot = -self.rotational_thrust

        self.apply_acceleration(lin, rot)

        reward, done = self.check_target()
        truncated = done

        observation = self.get_obs()
        return observation, reward, done, truncated, {}

    def check_target(self):
        distance = self.wrapped_distance(self.jet_position, self.target_position)
        if distance < 20:
            r, d = 1, True
            # self.target_position = np.random.uniform(0, self.dim, 2)
        else:
            r, d = -1 - (distance / self.dim), False
        return r, d

    def apply_acceleration(self, linear, rotational):
        self.linear_velocity *= self.linear_damping
        self.rotational_velocity *= self.rotation_damping

        if linear != 0:
            rad_angle = math.radians(self.jet_angle)
            self.linear_velocity += linear * np.array([math.cos(rad_angle), math.sin(rad_angle)])
            speed = np.linalg.norm(self.linear_velocity)
            if speed > self.max_linear_velocity:
                overspeed = speed / self.max_linear_velocity
                self.linear_velocity /= overspeed

        self.jet_position += self.linear_velocity
        self.jet_position %= self.grid_size
        self.target_position += self.target_velocity
        self.target_position %= self.grid_size

        if rotational != 0:
            self.rotational_velocity += rotational
            self.rotational_velocity = np.clip(
                self.rotational_velocity, -self.max_rotation_velocity, self.max_rotation_velocity
            )
        self.jet_angle += self.rotational_velocity
        self.jet_angle %= 360

    def render(self):
        if self.screen is None:
            # Pygame initialization
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

        jet_size = 15
        cos = lambda x: math.cos(math.radians(x))
        sin = lambda x: math.sin(math.radians(x))
        jet_points = [
            (
                self.jet_position[0] + jet_size * cos(self.jet_angle),
                self.jet_position[1] + jet_size * sin(self.jet_angle),
            ),
            (
                self.jet_position[0] + jet_size * cos(self.jet_angle + 140),
                self.jet_position[1] + jet_size * sin(self.jet_angle + 140),
            ),
            (
                self.jet_position[0] + jet_size * cos(self.jet_angle - 140),
                self.jet_position[1] + jet_size * sin(self.jet_angle - 140),
            ),
        ]
        if self.just_fired:
            self.shot_time = 10

        if self.shot_time > 0:
            self.shot_time -= 1
            pygame.draw.polygon(self.screen, RED, jet_points)
        else:
            pygame.draw.polygon(self.screen, WHITE, jet_points)

        target_radius = 10
        pygame.draw.circle(
            self.screen,
            RED,
            (
                int(self.target_position[0] * self.window_size[0] / self.grid_size[0]),
                int(self.target_position[1] * self.window_size[1] / self.grid_size[1]),
            ),
            target_radius,
        )

        pygame.display.flip()

    def close(self):
        # pygame.quit()
        return
