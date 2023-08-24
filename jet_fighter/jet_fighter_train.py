from jet_fighter_env import JetFighterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, VecFrameStack
import torch as th

# env = JetFighterEnv()
env = DummyVecEnv([JetFighterEnv, JetFighterEnv])
env = VecFrameStack(env, 4)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
env = VecMonitor(env)
# Define and Train the agent

print(env.observation_space)
exit()
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboard_logs/")
# model = PPO.load("./jet_moving_target", env)

for epoch in range(0, 10):
    model.learn(total_timesteps=100_000, tb_log_name="jet_stacked_frames", reset_num_timesteps=epoch == 0)
    

model.save("./jet_stacked_frames")
