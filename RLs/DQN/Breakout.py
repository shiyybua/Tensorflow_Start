# -*- coding: utf-8 -*
import sys
sys.path.append('/home/caiww/code')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import gym
import tensorflow as tf
from DQN.Brain import DQN
import time
from state_processor import StateProcessor

env = gym.make('Breakout-v0')
env.seed(1)     # reproducible.
env = env.unwrapped
epoch = 1000

print(env.action_space)
print(env.observation_space)

RESOURCE_PATH = '../resource/DQN/'
train = True
is_gpu_available = None #otherwise: e.g. ['/gpu:2', '/gpu:3']

if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        Q_net = DQN("Q_net",sess,is_gpu_available=is_gpu_available)
        target_net = DQN("target_net",sess,is_gpu_available=is_gpu_available)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(RESOURCE_PATH)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        train_writer = tf.summary.FileWriter(RESOURCE_PATH + 'DQN_record',
                                             sess.graph)
        step = 0
        merged = tf.summary.merge_all()

        image_processor = StateProcessor()
        for episode in range(epoch):
            start = time.time()
            observation = env.reset()
            observation = image_processor.process(observation)
            if train and episode % 100 == 0 and episode != 0:
                saver.save(sess, RESOURCE_PATH + 'DQN_checkpoints', global_step=episode)

            while True:
                # fresh env
                # env.render()

                action = Q_net.choose_action(observation)
                observation_, reward, done, _ = env.step(action)

                observation_ = image_processor.process(observation_)
                Q_net.store_transition(observation,action,reward,observation_)
                observation = observation_

                if train and (step > 200) and (step % 20 == 0):
                    Q_net.learn(target_net, merged, train_writer)

                step += 1

                if done:
                    print episode
                    end = time.time()
                    print end - start
                    print
                    break



