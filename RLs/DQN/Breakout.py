# -*- coding: utf-8 -*
import sys
sys.path.append('/Users/mac/PycharmProjects/Tensorflow_Start')
import gym
import tensorflow as tf
from RLs.DQN.Brain import DQN

env = gym.make('Breakout-v0')
env.seed(1)     # reproducible.
env = env.unwrapped
epoch = 1000

print(env.action_space)
print(env.observation_space)

if __name__ == '__main__':
    with tf.Session() as sess:
        Q_net = DQN("Q_net",sess)
        target_net = DQN("target_net",sess)
        train_writer = tf.summary.FileWriter('../../resource/DQN_record',
                                             sess.graph)
        step = 0
        merged = tf.summary.merge_all()
        for episode in range(epoch):
            observation = env.reset()
            while True:
                # fresh env
                env.render()
                action = Q_net.choose_action(observation)
                observation_, reward, done, _ = env.step(action)
                Q_net.store_transition(observation,action,reward,observation_)

                if (step > 200) and (step % 5 == 0):
                    Q_net.learn(target_net, merged, train_writer)


                step += 1

                if done:
                    break

            print episode


