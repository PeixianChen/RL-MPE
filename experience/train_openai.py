import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_world_openai", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=4, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/openai/model/maddpg_maddpg/", help="directory in which training state and model should be saved")
    parser.add_argument("--summary-dir", type=str, default="./tmp/openai/summary/maddpg_maddpg/")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def lstm_mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions

    lstm_size = input.shape[1]
    input = tf.expand_dims(input,0)# [1,?,232]

    with tf.variable_scope(scope, reuse=reuse):
        # fully_connetcted: 全连接层
        out = input
        lstm = rnn.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm.zero_state(1, dtype=tf.float32)
        # outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(lstm, out, time_major=False, initial_state = init_state)
        # outputs = tf.convert_to_tensor(np.array(outputs))
        out = layers.fully_connected(outputs[-1], num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model_adv = mlp_model
    model_good = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
           "agent_%d" % i, model_adv, obs_shape_n, env.action_space, i, arglist,
           local_q_func=(arglist.adv_policy=='ddpg')))
        #trainers.append(trainer(
         #   "agent_%d" % i, mlp_model, obs_shape_n, env.action_space, i, arglist,
          #  local_q_func=arglist.adv_policy))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
           "agent_%d" % i, model_good, obs_shape_n, env.action_space, i, arglist,
           local_q_func=(arglist.good_policy=='ddpg')))
        #trainers.append(trainer(
         #   "agent_%d" % i, mlp_model, obs_shape_n, env.action_space, i, arglist,
          #  local_q_func=arglist.good_policy))
    return trainers


def setup_summary():
    RedAgent_total_reward_episode = tf.Variable(0.0)
    tf.summary.scalar("mlp_RedAgent_reward/episode", RedAgent_total_reward_episode)
    episode_duration_episode = tf.Variable(0.0)
    tf.summary.scalar("Duration/episode", episode_duration_episode)
    GreenAgent_total_reward_episode = tf.Variable(0.0)
    tf.summary.scalar("mlp_GreenAgent_reward/episode", GreenAgent_total_reward_episode)

    summary_vars = [RedAgent_total_reward_episode, episode_duration_episode, GreenAgent_total_reward_episode]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op


def train(arglist):
    with U.single_threaded_session() as sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        saver = tf.train.Saver()
        # Initialize
        U.initialize()
        summary_writer = tf.summary.FileWriter(arglist.summary_dir, sess.graph)
        summary_placeholders, update_ops, summary_op = setup_summary()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
            #saver.restore(sess, "/home/sugon/Peixian/maddpg_peixian/maddpg/experiments/tmp/policy/simple_comm_-4166440")
            #print ("susessfully restor")

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver(max_to_keep=3)
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        adversary_rewards = 0.0
        goodagent_rewards = 0.0

        print('Starting iterations...')
        while True:
            #input('...')
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

         
            for i, rew in enumerate(rew_n):
                #print (i,":",rew_n[i])
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
                if i < num_adversaries:
                    adversary_rewards += rew
                else:
                    goodagent_rewards += rew
           


            if done or terminal:
                if done:
                    print("*"*20)
                    print ("done:",episode_step)


                stats = [adversary_rewards, episode_step, goodagent_rewards]
                for i in range(len(stats)):
                    sess.run(update_ops[i], feed_dict={
                        summary_placeholders[i]: float(stats[i])
                    })
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, len(episode_rewards)+1)

                obs_n = env.reset()
                episode_step = 0
                adversary_rewards = 0.0
                goodagent_rewards = 0.0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, train_step, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
