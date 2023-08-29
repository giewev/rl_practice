from jet_fighter_env import JetFighterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import numpy as np
import itertools
from multi_agent import train_action_mask

def train(model_name, stack_count, scale, topology, steps = 1_000_000, sub_envs = 4, load = False):
    env = DummyVecEnv([JetFighterEnv for x in range(sub_envs)])
    env = VecFrameStack(env, stack_count)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    env = VecMonitor(env)
    
    callbacks = [
        CheckpointCallback(
        save_freq=100_000,
        save_path="./model_snapshots/",
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
        ),
        # TensorboardCallback()
    ]

    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                        net_arch=np.array(topology) * scale)
    
    if load:
        model = PPO.load(f"./{model_name}", env)
    else:
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboard_logs/")


    model.learn(total_timesteps=steps, tb_log_name=model_name, reset_num_timesteps=not load, callback=callbacks)
    model.save(f"./{model_name}")

if __name__ == '__main__':
    # stack_counts = [1,4,16]
    # network_scales = [8, 32, 64]
    # network_topologies = [[4,2,1], [8, 4, 2, 1], [2,2]]
    # activations = [th.nn.Tanh, th.nn.ReLU]
    # plans = itertools.product(stack_counts, network_scales, network_topologies)
    # for stacks, scale, topology in plans:
    #     train()

    train(f'')
