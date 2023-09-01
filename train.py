from stable_baselines3 import PPO, DDPG, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from pettingzoo.mpe import simple_world_comm_v3, simple_speaker_listener_v4
import torch as th
import numpy as np

from envs import OvermindEnv

if __name__ == "__main__":
    model_name = "overmind_2_targets"

    env = OvermindEnv.build_with_wrappers(
        num_workers=2,
        num_pointers=2,
        num_targets=2,
        num_envs=4,
        num_processes=4,
        num_frames=4,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=(100_000) // env.num_envs // len(env.possible_agents),
            save_path="./model_snapshots/",
            name_prefix=model_name,
            save_replay_buffer=True,
            save_vecnormalize=True,
        ),
        # TensorboardCallback()
    ]

    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=np.array([128, 128]))

    load = False
    if load:
        model = PPO.load(f"./saved_models/{model_name}", env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )

    model.learn(
        total_timesteps=20_000_000,
        tb_log_name=model_name,
        reset_num_timesteps=not load,
        callback=callbacks,
    )
    model.save(f"./saved_models/{model_name}")
    env.close()
