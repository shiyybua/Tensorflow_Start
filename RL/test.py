# -*- coding: utf-8 -*
import gym

def cart_pole():
    env = gym.make('CartPole-v1')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

def CarRacing():
    # https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md 源码安装2DBOX
    env = gym.make('CarRacing-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


CarRacing()