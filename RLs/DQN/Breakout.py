import gym
import tensorflow as tf
import numpy as np
from .Brain import DQN

env = gym.make('Breakout-v0')
env.seed(1)     # reproducible.
env = env.unwrapped
epoch = 1000

print(env.action_space)
print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

# while True:
#     env.render()
#     action = 1
#     observation_, reward, done, info = env.step(action)
#     print observation_
#     print reward
#     break


with tf.Session() as sess:
    Q_net = DQN("Q_net",sess)
    target_net = DQN("target_net",sess)
    step = 0
    for episode in range(epoch):
        observation = env.reset()
        while True:
            # fresh env
            env.render()
            action = Q_net.choose_action(observation)
            observation_, reward, done = env.step(action)
            Q_net.store_transition(observation,action,reward,observation_)

            if (step > 200) and (step % 5 == 0):
                Q_net.learn()

            step += 1

            if done:
                break






