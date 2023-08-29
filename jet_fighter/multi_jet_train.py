from pettingzoo.test import parallel_api_test
from logger import TensorboardCallback
from multi_jet_env import MultiJetEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, VecFrameStack
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import numpy as np

if __name__ == "__main__":
    model_name = 'multi_jet_double_angle'
    num_envs = 4
    num_processes = 4

    # env = MultiJetEnv()
    env = aec_to_parallel(MultiJetEnv())
    env = ss.agent_indicator_v0(env)
    env = ss.frame_stack_v2(env, stack_size=16)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=num_processes, base_class="stable_baselines3")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    env = VecMonitor(env)

    # parallel_api_test(env, num_cycles=1_000)

    callbacks = [
        CheckpointCallback(
            save_freq=100_000 / num_envs,
            save_path="./model_snapshots/",
            name_prefix=model_name,
            save_replay_buffer=True,
            save_vecnormalize=True,
            ),
        TensorboardCallback()
    ]

    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                        net_arch=np.array([64,64]))
    
    # env.reset(seed=42)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1
    )

    load = False
    if load:
        model = PPO.load(f"./saved_models/{model_name}", env)
    else:
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboard_logs/")

    model.learn(total_timesteps=5_000_000, tb_log_name=model_name, reset_num_timesteps=not load, callback=callbacks)
    model.save(f"./saved_models/{model_name}")
    env.close()
