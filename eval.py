# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import codecs
import copy
import os

import librosa
from scipy.io.wavfile import write

from hyperparams import Hyperparams as hp
import numpy as np
from prepro import *
import tensorflow as tf
from train import Graph
from utils import *


def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X = load_eval_data() # texts
    char2idx, idx2char = load_vocab()
             
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
             
            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name


            timesteps = 700  # Adjust this number as you want
            outputs1 = np.zeros((hp.num_samples, timesteps, hp.n_mels), np.float32) 
            # outputs1 = np.zeros((hp.num_samples, timesteps, hp.n_mels * hp.r), np.float32)  # hp.n_mels*hp.r
            for j in range(int(timesteps//hp.r)-2*hp.r):
                _outputs1 = sess.run(g.outputs1, {g.x: X, g.y: outputs1})
                for k in range(hp.r):
                    outputs1[:,hp.r*j+k,:] = _outputs1[:,hp.r*j+k,:]

            outputs1 = np.reshape(outputs1,(X.shape[0],-1,hp.n_mels))
            outputs2 = sess.run(g.outputs2, {g.outputs1: outputs1})

    # Generate wav files
    if not os.path.exists(hp.outputdir): os.mkdir(hp.outputdir) 
    with codecs.open(hp.outputdir + '/text.txt', 'w', 'utf-8') as fout:
        for i, (x, mag) in enumerate(zip(X, outputs2)):
            # write text
            fout.write(str(i) + "\t" + "".join(idx2char[idx] for idx in np.fromstring(x, np.int32) if idx != 0) + "\n")
            
            #s = restore_shape(s, hp.win_length//hp.hop_length, hp.r)
                         
            # generate wav files
            audio = spectrogram2wav(mag)
            # else:
            #     s = np.where(s < 0, 0, s)
            #     audio = spectrogram2wav(s**hp.power)
            write(hp.outputdir + "/{}_{}.wav".format(mname, i), hp.sr, audio)     
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
