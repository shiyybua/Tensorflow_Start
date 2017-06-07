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

Q_net = DQN("Q_net")
target_net = DQN("target_net")
memory = np.array()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for episode in range(epoch):
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # # RL choose action based on observation
            # action = RL.choose_action(observation)
            #
            # # RL take action and get next observation and reward
            # observation_, reward, done = env.step(action)



