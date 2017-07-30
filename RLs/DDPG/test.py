import tensorflow as tf
import gym

ENV_NAME = 'Pong-v0'

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space
a_dim = env.action_space

print s_dim
print a_dim