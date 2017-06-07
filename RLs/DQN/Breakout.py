import gym

env = gym.make('Breakout-v0')
env.seed(1)     # reproducible.
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

while True:
    env.render()
    action = 1
    observation_, reward, done, info = env.step(action)
    print observation_
    print reward
    break

