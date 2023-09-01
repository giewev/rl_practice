import gymnasium as gym
import numpy as np
import math
import pygame
from pettingzoo import AECEnv
import functools
from pettingzoo.utils import agent_selector
from physics import Runner, Pather
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel


class OvermindEnv(AECEnv):
    """Overmind Environment that follows gym interface."""

    metadata = {"name": "Overmind", "is_parallelizable": True, "render_mode": "human"}

    @staticmethod
    def build_with_wrappers(
        num_workers=1,
        num_pointers=1,
        num_targets=1,
        num_envs=1,
        num_processes=1,
        num_frames=1,
    ):
        env = OvermindEnv(
            num_workers=num_workers, num_pointers=num_pointers, num_targets=num_targets
        )
        env = aec_to_parallel(env)
        env = ss.pad_observations_v0(env)
        env = ss.agent_indicator_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.frame_stack_v2(env, stack_size=num_frames)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(
            env, num_envs, num_cpus=num_processes, base_class="stable_baselines3"
        )
        env = VecNormalize(
            env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99
        )
        env = VecMonitor(env)
        return env

    def __init__(self, num_workers=1, num_pointers=1, num_targets=1):
        super().__init__()
        self.render_mode = "human"
        self.dim = 1000
        self.grid_size = (self.dim, self.dim)
        self.past_pointers = []

        self.num_workers = num_workers
        self.num_pointers = num_pointers
        self.num_targets = num_targets

        self.possible_agents = ["overmind_0"] + [
            f"worker_{x+1}" for x in range(num_workers)
        ]
        self.agents = self.possible_agents[:]
        self.screen = None
        self.seed_value = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        worker_pointer_coords = 2 * self.num_workers * self.num_pointers
        if "overmind" in agent:
            return gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(worker_pointer_coords + self.num_targets * 2,),
                dtype=np.float32,
            )
        elif "worker" in agent:
            return gym.spaces.Box(
                low=-1.0, high=1.0, shape=(worker_pointer_coords,), dtype=np.float32
            )
        raise ValueError(f"Idk what {agent} is")

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if "overmind" in agent:
            return gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2 * self.num_pointers,), dtype=float
            )
        elif "worker" in agent:
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        raise ValueError(f"Idk what {agent} is")

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
        self.rewards = {x: 0.0 for x in self.agents}
        self.infos = {x: {} for x in self.agents}

        self.targets = [Pather(self.grid_size, 20) for x in range(self.num_targets)]
        self.pointers = np.zeros((self.num_pointers * 2,))

        self.workers = {}
        for x in self.agents:
            if "worker" in x:
                self.workers[x] = Runner(self.grid_size, 30, wrap=False)
                self.workers[x].randomize_position()
                self.workers[x].randomize_heading()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.get_all_obs(), {}

    def normalize_wrapped_position(self, pos):
        return self.normalize_cyclic(pos, self.dim)

    def normalize_position(self, pos):
        return self.normalize_scalar(pos, 0, self.dim)

    def normalize_relative_position(self, rel):
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
        return (2 * (x - low) / (high - low)) - 1

    def worker_positions_normalized(self):
        return [
            self.normalize_position(self.workers[x].position)
            for x in sorted(self.workers.keys())
        ]

    def pointers_normalized(self):
        return [self.pointers]

    def targets_normalized(self):
        return [self.normalize_position(x.position) for x in self.targets]

    def pointer_relative_positions_normalized(self):
        positions = []
        for name in sorted(self.workers.keys()):
            for x in range(0, self.num_pointers * 2, 2):
                rel = (
                    self.pointer_to_pos(self.pointers[x : x + 2])
                    - self.workers[name].position
                )
                positions.append(self.normalize_relative_position(rel))
        return positions

    def observe(self, agent):
        if "overmind" in agent:
            # print(self.worker_positions_normalized())
            # print(self.targets_normalized())
            return np.concatenate(
                self.worker_positions_normalized() + self.targets_normalized()
            )
        elif "worker" in agent:
            return np.concatenate(self.pointer_relative_positions_normalized())
        return f"Idk what {agent} is"

    def get_all_obs(self):
        return self.for_all_agents(self.observe)

    def for_all_agents(self, f):
        return {x: f(x) for x in self.agents}

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0

        if "overmind" in agent:
            for x in self.targets:
                x.update()
            self.past_pointers.append(self.pointers)
            self.past_pointers = self.past_pointers[-4:]
            self.pointers = action
        elif "worker" in agent:
            self.workers[agent].choose_direction(action)
            self.workers[agent].update()
        else:
            raise NameError(f"Idk what {agent} is")

        if self._agent_selector.is_last():
            self.steps += 1
            self.remaining_steps -= 1
            self.proximity_reward()
            done = self.out_of_bounds() or self.remaining_steps <= 0
            # self.speed_bonus()
            self.terminations = self.for_all_agents(lambda x: done)
            self.truncations = self.for_all_agents(lambda x: done)
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def proximity_reward(self):
        reward = 0
        for target in self.targets:
            distance = 99999
            for name, worker in self.workers.items():
                distance = min(
                    distance, np.linalg.norm(worker.position - target.position)
                )
            reward -= distance / self.dim
        self.rewards = self.for_all_agents(lambda x: reward)

    def speed_bonus(self):
        reward = 0
        for name, worker in self.workers.items():
            reward += np.linalg.norm(worker.linear_velocity) / 100
        self.rewards = self.for_all_agents(lambda x: self.rewards[x] + reward)

    def out_of_bounds(self):
        for name, worker in self.workers.items():
            if worker.out_of_bounds():
                self.rewards = self.for_all_agents(
                    lambda x: -self.remaining_steps * self.num_targets * 2
                )
                return True
        return False

    def pointer_to_pos(self, pointer):
        return (pointer + 1) * self.dim / 2

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
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)

        self.screen.fill(BLACK)
        for x in self.workers.values():
            x.render(self.screen, WHITE)
        for x in range(0, self.num_pointers * 2, 2):
            for p in [self.pointers] + self.past_pointers:
                position = self.pointer_to_pos(p[x : x + 2])
                pygame.draw.circle(
                    self.screen,
                    RED,
                    (
                        int(position[0]),
                        int(position[1]),
                    ),
                    10,
                )
        for x in self.targets:
            pygame.draw.circle(
                self.screen,
                GREEN,
                (
                    int(x.position[0]),
                    int(x.position[1]),
                ),
                10,
            )
        # self.jets['mouse'].render(self.screen, WHITE)
        # self.target.render(self.screen, RED)

        pygame.display.flip()

    def close(self):
        # pygame.quit()
        return
