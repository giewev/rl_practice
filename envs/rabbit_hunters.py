from locale import normalize
import gymnasium as gym
import numpy as np
import math
import pygame
from physics import PhysicsObject, Jet
from pettingzoo import AECEnv
from pettingzoo import ParallelEnv
import functools
from pettingzoo.utils import agent_selector

class RabbitHuntersEnv(AECEnv):
    """Jet Fighter Environment that follows gym interface."""

    metadata = {'name' : 'jet_fighter', 'is_parallelizable': True}

    def __init__(self):
        super().__init__()
        self.render_mode = "human"
        self.dim = 1000
        self.grid_size = (self.dim, self.dim)

        self.possible_agents = ['mouse', 'cat']
        self.agents = self.possible_agents[:]
        self.screen = None
        self.seed_value = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # return gym.spaces.MultiDiscrete([2, 3])
        return gym.spaces.Box(np.array([-1, -1]), 1, dtype = float)

    def reset(self, seed=None, options=None):
        self.seed_value = seed
        np.random.seed(seed)

        self.max_steps = 1000
        self.steps = 0
        self.remaining_steps = self.max_steps

        self.agents = self.possible_agents[:]
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {x:0.0 for x in self.agents}
        self.infos = {x:{} for x in self.agents}

        self.jets = {}
        for x in self.agents:
            self.jets[x] = Jet(self.grid_size, wrap = False)
            self.jets[x].randomize_position()
            # self.jets[x].randomize_velocity()
            self.jets[x].randomize_heading()
            # self.jets[x].randomize_angular_velocity()
        
        self.jets['mouse'].max_linear_velocity = 40

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
 
        return self.get_all_obs(), {}

    def normalize_wrapped_position(self, pos):
        return self.normalize_cyclic(pos, self.dim)
    
    def normalize_position(self, pos):
        return self.normalize_scalar(pos, 0, self.dim)
    
    def normalize_relative_position(self,rel):
        return self.normalize_scalar(rel, -self.dim, self.dim)

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
    
    def observe(self, agent):
        active = self.jets[agent]
        other = next(self.jets[x] for x in self.jets if x != agent)

        return np.concatenate(
            [
                self.normalize_position(active.position), # 2
                self.normalize_position(other.position), # 2
                self.normalize_relative_position(other.position - active.position), # 2
                # self.normalize_rads(active.heading), # 2
                # self.normalize_rads(other.heading), # 2
                # self.normalize_rads(active.relative_bearing_of(other)) # 2
            ]
        )
    
    def get_all_obs(self):
        return self.for_all_agents(self.observe)

    def get_enemy(self, agent):
        if agent == 'cat':
            return 'mouse'
        return 'cat'

    def for_all_agents(self, f):
        return {x:f(x) for x in self.agents}

    def step(self, action):
        self.steps += 1
        self.remaining_steps -= 1

        agent = self.agent_selection
        current_jet = self.jets[agent]
        enemy_agent = self.get_enemy(agent)
        enemy_jet = self.jets[enemy_agent]

        if (self.terminations[agent] or self.truncations[agent]):
            self._was_dead_step(action)
            return
        
        self._cumulative_rewards[agent] = 0
        
        # current_jet.thrust(action[0])
        # current_jet.heading += np.pi * action[1] / 10
        current_jet.linear_velocity = action * current_jet.max_linear_velocity
        current_jet.heading = np.arctan2(action[1], action[0])
        current_jet.update()

        if self._agent_selector.is_last():
            if self.check_jet_collision():
                self.cat_wins()
            elif self.jets['mouse'].out_of_bounds():
                self.cat_wins()
            elif self.jets['cat'].out_of_bounds():
                self.mouse_wins()
            elif self.remaining_steps == 0:
                self.mouse_wins()
            else:
                self.no_winner()
        else:
            self._clear_rewards()
        
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        # obs = self.get_all_obs()
        # agent = self.get_enemy(agent)
        # infos = self.for_all_agents(lambda x: {})

        # if done['cat']:
        #     self.agents = []

        # return obs, self.rewards, terminations, truncations, infos

    def mouse_wins(self):
        self.rewards = {'cat' : -self.remaining_steps, 'mouse' : 0}
        self.terminations = self.for_all_agents(lambda x: True)
        self.truncations = self.for_all_agents(lambda x: True)

    def cat_wins(self):
        self.rewards = {'cat' : self.remaining_steps, 'mouse' : -self.remaining_steps}
        self.terminations = self.for_all_agents(lambda x: True)
        self.truncations = self.for_all_agents(lambda x: True)
    
    def no_winner(self):
        self.rewards = {'cat' : -1, 'mouse' : 1}
        self.terminations = self.for_all_agents(lambda x: False)
        self.truncations = self.for_all_agents(lambda x: False)

    def check_jet_collision(self):
        distance = self.jets['mouse'].distance(self.jets['cat'])
        if distance < 30:
            return True
    
    def render(self):
        if self.screen is None:
            print("initializing screen")
            self.window_size = (self.dim, self.dim)
            self.screen = pygame.display.set_mode(self.window_size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)

        self.screen.fill(BLACK)
        self.jets['cat'].render(self.screen, RED)
        self.jets['mouse'].render(self.screen, WHITE)
        # self.target.render(self.screen, RED)

        pygame.display.flip()

    def close(self):
        # pygame.quit()
        return
