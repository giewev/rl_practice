from jet_fighter_env import JetFighterEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, VecFrameStack
import pygame
import time
import supersuit as ss
from multi_jet_env import MultiJetEnv
from pettingzoo.utils.conversions import aec_to_parallel 
import numpy as np

if __name__ == '__main__':
    pygame.init()

    # env = JetFighterEnv()
    env = aec_to_parallel(MultiJetEnv())
    env = ss.agent_indicator_v0(env)
    env = ss.frame_stack_v2(env, stack_size=16)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)
    # env = VecMonitor(env)
    
    model_name = 'multi_jet_double_angle'
    model = PPO.load(f"./saved_models/{model_name}", env)
    # model = PPO.load(f"./model_snapshots/{model_name}", env)
    clock = pygame.time.Clock()

    # Test the trained agent
    # using the vecenv
    obs = env.reset()
    n_steps = 10000
    steps_to_catch = 0
    for step in range(n_steps):
        # env.action_space(env.possible_agents[0]).seed(i)
        act = model.predict(obs, deterministic=True)[0]
        # act[0] = env.action_space.sample()
        # act[1] = 0

        # print(act)
        # if step % 2 == 1:
        #     act = model.predict(obs, deterministic=True)[0]
        # else:
        #     act = env.action_space.sample()
        #     # act = np.array([[0,0], [act[0], act[1]]])
        #     act = np.array([[act[0], act[1]], [0,0]])
        obs, reward, done, info = env.step(act)
        # obs, reward, termination, truncation, info = env.step()

        # for agent in env.agents:
        #     rewards[agent] += env.rewards[agent]

        if done.any():
            env.reset()
            print(f"It took {steps_to_catch} steps for the cat to catch the mouse")
            steps_to_catch = 0
        else:
            steps_to_catch += 1
        env.render()
        clock.tick(30)

    # env.close()
