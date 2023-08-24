from jet_fighter_env import JetFighterEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
import pygame
import time

pygame.init()

# env = JetFighterEnv()
env = DummyVecEnv([JetFighterEnv])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)
env = VecMonitor(env)

model = PPO.load("./jet_moving_target", env)
clock = pygame.time.Clock()

# Test the trained agent
# using the vecenv
obs = env.reset()
n_steps = 10000
for step in range(n_steps):
    # print(obs)
    action, _ = model.predict(obs, deterministic=True)
    # action = [model.action_space.sample()]
    # print(action)
    print(f"Step {step + 1}")
    # print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print(f"Reward {reward}")
    # print("obs=", obs, "reward=", reward, "done=", done)
    if done:
        print("Goal reached!", "reward=", reward)
        # obs = env.reset()
    else:
        env.render(mode="human")
        clock.tick(30)
        # pygame.display.update()
    # time.sleep(1 / 30)

# env.close()
