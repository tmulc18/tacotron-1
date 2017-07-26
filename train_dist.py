# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import os
import argparse
import sys

import librosa
from tqdm import tqdm

from data_load import get_batch
from hyperparams import Hyperparams as hp
from modules import *
from networks import encode, decode1, decode2
import numpy as np
from prepro import *
from prepro import load_vocab
import tensorflow as tf
from utils import shift_by_one, byte_size_load_fn
import time

FLAGS = None

class Graph:
    # Load vocabulary 
    char2idx, idx2char = load_vocab()
    
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            n_ps = len(hp.cluster_spec['ps'])
            greedy=tf.contrib.training.GreedyLoadBalancingStrategy(num_tasks=n_ps,load_fn=byte_size_load_fn)
            with tf.device(tf.train.replica_device_setter(ps_tasks=len(hp.cluster_spec['ps'])\
                ,worker_device="/job:worker/task:%d" % FLAGS.task_index,ps_strategy=greedy)):
                if is_training:
                    self.x, self.y, self.z, self.num_batch = get_batch()
                else: # Evaluation
                    self.x = tf.placeholder(tf.int32, shape=(None, None))
                    self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))

                self.decoder_inputs = shift_by_one(self.y)
                
                with tf.variable_scope("net"):
                    # Encoder
                    self.memory = encode(self.x, is_training=is_training) # (N, T, E)
                    
                    # Decoder 
                    self.outputs1 = decode1(self.decoder_inputs, 
                                         self.memory,
                                         is_training=is_training) # (N, T', hp.n_mels*hp.r)
                    self.outputs2 = decode2(self.outputs1, is_training=is_training) # (N, T', (1+hp.n_fft//2)*hp.r)
                 
                if is_training:
                    # Loss
                    if hp.loss_type=="l1": # L1 loss
                        self.loss1 = tf.abs(self.outputs1 - self.y)
                        self.loss2 = tf.abs(self.outputs2 - self.z)
                    else: # L2 loss
                        self.loss1 = tf.squared_difference(self.outputs1, self.y)
                        self.loss2 = tf.squared_difference(self.outputs2, self.z)
                    
                    # Target masking
                    if hp.target_zeros_masking:
                        self.loss1 *= tf.to_float(tf.not_equal(self.y, 0.))
                        self.loss2 *= tf.to_float(tf.not_equal(self.z, 0.))
                    
                    self.mean_loss1 = tf.reduce_mean(self.loss1)
                    self.mean_loss2 = tf.reduce_mean(self.loss2)
                    self.mean_loss = self.mean_loss1 + self.mean_loss2 
                         
                                                    
                    # Training Scheme
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    #self.global_step = tf.contrib.framework.get_or_create_global_step()
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

                    # Synchronous
                    if hp.synch:
                        self.optimizer = tf.train.SyncReplicasOptimizer(self.optimizer, 
                                                replicas_to_aggregate=len(hp.ips)*hp.n-hp.n_stale,
                                                total_num_replicas=len(hp.ips)*hp.n)


                    #gradient clipping
                    grads,var_list = zip(*self.optimizer.compute_gradients(self.mean_loss))
                    grads_clipped,_=tf.clip_by_global_norm(grads,5.)
                    self.train_op= self.optimizer.apply_gradients(zip(grads_clipped,var_list),global_step=self.global_step)
                    #self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)


                    # Summmary 
                    tf.summary.scalar('mean_loss1', self.mean_loss1)
                    tf.summary.scalar('mean_loss2', self.mean_loss2)
                    tf.summary.scalar('mean_loss', self.mean_loss)
                    
                    self.merged = tf.summary.merge_all()

                    # For distributed
                    self.settle_step = tf.Variable(0, name='settle_step', trainable=False)
                    one = tf.constant(1)
                    self.inc_settle = tf.assign_add(self.settle_step,one)

                    if hp.synch:
                        self.init = tf.global_variables_initializer()
                        self.saver = tf.train.Saver()
         
def main():
    cluster = tf.train.ClusterSpec(hp.cluster_spec) #lets this node know about all other nodes
    if FLAGS.job_name == "ps":
        with tf.device('/cpu:0'):
            server = tf.train.Server(cluster,job_name="ps",task_index=FLAGS.task_index)
            server.join()
    else:
        hp.wkr = FLAGS.task_index

        is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
        gpu_options = tf.GPUOptions(allow_growth=True,allocator_type="",visible_device_list="%d"%FLAGS.task_index)
        config = tf.ConfigProto(allow_soft_placement=True,device_count={'GPU':1},inter_op_parallelism_threads=0,intra_op_parallelism_threads=0)  
        server = tf.train.Server(cluster,job_name="worker",
                        task_index=FLAGS.task_index,config=config)
        
        g = Graph(); print("Training Graph loaded")
        
        with g.graph.as_default():
            # Load vocabulary 
            char2idx, idx2char = load_vocab()
            
            # Training 
            # sv = tf.train.Supervisor(logdir=hp.logdir,
            #                          save_model_secs=600,is_chief=is_chief)

            # GPU settings
            #gpu_options = tf.GPUOptions(allow_growth=True,allocator_type="BFC") # try to remove
            #config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,inter_op_parallelism_threads=1,intra_op_parallelism_threads=1) #try to remove

            if hp.synch:
                # Hooks
                sync_replicas_hook = g.optimizer.make_session_run_hook(is_chief)
                scaff=tf.train.Scaffold(init_op=g.init,saver=g.saver,summary_op=g.merged)
                saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=hp.logdir,scaffold=scaff,save_steps=1000)
                summary_hook = tf.train.SummarySaverHook(save_steps=10,output_dir=hp.logdir,scaffold=scaff)
                step_cnt_hook=tf.train.StepCounterHook(output_dir=hp.logdir)
                hooks=[sync_replicas_hook,summary_hook,saver_hook,step_cnt_hook]

                # Monitored Session
                sess = tf.train.MonitoredTrainingSession(server.target,is_chief=is_chief,
                                                        config=config,hooks=hooks,
                                                        checkpoint_dir=hp.logdir,scaffold=scaff)
            else:
                sv = tf.train.Supervisor(logdir=hp.logdir,
                                     save_model_secs=600,is_chief=is_chief)
                sess = sv.prepare_or_wait_for_session(server.target,config=config,start_standard_services=True)
            ss = sess.run(g.settle_step)
            for epoch in range(1, hp.num_epochs+1):
                if is_chief:
                    gs = sess.run(g.global_step) 
                    #sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
                    #sv.start_queue_runners(sess, )

                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b%d'%FLAGS.task_index):
                    if hp.synch:
                        if sess.should_stop(): print('fuck'); break
                    else:
                        if sv.should_stop(): print('fuck'); break

                    # Synchronous
                    if hp.synch:
                        sess.run(g.train_op)

                    # Asynchronous
                    else:
                        if ss <= hp.settle_steps*len(hp.worker):
                            if is_chief:
                                sess.run([g.train_op,g.inc_settle])
                                ss = sess.run(g.settle_step)
                            else:
                                while(ss<hp.settle_steps*FLAGS.task_index):
                                    time.sleep(.01)
                                    ss = sess.run(g.settle_step)
                        else:
                            sess.run(g.train_op)

                    # Create the summary every 100 chief steps.
                    #sv.summary_computed(sess, sess.run(g.merged))
                
                # # Write checkpoint files at every epoch
                # if is_chief:
                #     gs = sess.run(g.global_step) 
                #     sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
    print("Done")
