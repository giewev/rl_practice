from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from collections import defaultdict
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.agent_rewards = defaultdict(list)

    def _on_step(self) -> bool:
        # print(self.locals)
        # exit()
        for i, reward in enumerate(self.locals['rewards']):
            agent = ['mouse', 'cat'][i%2]
            self.agent_rewards[agent].append(reward)
        # value = np.random.randint(0,10, 2)
        # print(self.locals['actions'])
        # actions = self.locals['actions']
        # self.thrust_actions.append(actions[0][0])
        # self.rotate_actions.append(actions[0][1])
        return True
    
    def _on_rollout_end(self) -> None:
        # print(self.locals)
        freq = 50_000
        # if len(self.thrust_actions) > freq:
        #     self.logger.record("train/thrust_actions", th.tensor(self.thrust_actions))
        #     self.logger.record("train/rotate_actions", th.tensor(self.rotate_actions))

            # self.thrust_actions = []
            # self.rotate_actions = []
        for x in self.agent_rewards:
            self.logger.record(f"rollout/{x}_mean_reward", np.mean(self.agent_rewards[x]))
        self.agent_rewards = defaultdict(list)