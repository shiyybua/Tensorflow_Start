# -*- coding: utf-8 -*
import gym

env = gym.make('Breakout-v0')
env.seed(1)     # reproducible.
env = env.unwrapped
epoch = 1000

print(env.action_space)
print(env.observation_space)
# print(env.action_bound)

for episode in range(epoch):
    observation = env.reset()

    # if episode % 100 == 0 and episode != 0:
    #     saver.save(sess, RESOURCE_PATH + 'DQN_checkpoints', global_step=episode)
    # env.step(0)
    env.step(1)
    while True:
        # fresh env
        env.render()
        observation_, reward, done, info = env.step(0)

        observation = observation_
        print reward
        if done:
            break
    print episode







