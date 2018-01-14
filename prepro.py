#/usr/bin/python2
# -*- coding: utf-8 -*-

'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

import codecs
import csv
import os
import re

from hyperparams import Hyperparams as hp
import numpy as np

from cleaners import english_cleaners


def load_vocab():
    #vocab = "EG abcdefghijklmnopqrstuvwxyz'" # E: Empty. ignore G
    vocab = '_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;?"[] '
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    return char2idx, idx2char    

def create_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab() 
      
    lengths, texts, sound_files = [], [], []
    #reader = csv.reader(codecs.open(hp.text_file, 'rb', 'utf-8'))
    #for row in reader:

    if not hp.LJ:
        for row in codecs.open(hp.text_file, 'rb', 'utf-8'):
            print(row)
            sound_fname, text, duration = row.split("\t")
            sound_file = hp.sound_fpath + "/" + sound_fname + ".wav"
            #text = re.sub(r"[^ a-z']", "", text.strip().lower())
            text = english_cleaners(text)
             
            if hp.min_len <= len(text) <= hp.max_len:
                texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
                sound_files.append(sound_file)
    else:
        reader = csv.reader(codecs.open("LJSpeech-1.0/metadata.csv", 'rb', 'utf-8'))
        for line in codecs.open("LJSpeech-1.0/metadata.csv", 'r', 'utf-8'):
            sound_fname ,_ ,text = line.split('|')
            sound_file = "LJSpeech-1.0/wavs/" + sound_fname + ".wav"
            text = english_cleaners(text)
            length = len(text)
            lengths.append(length)
            texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
            sound_files.append(sound_file)
    return lengths, texts, sound_files
     
def load_train_data():
    """We train on the whole data but the last num_samples."""
    lengths, texts, sound_files = create_train_data()
    if hp.sanity_check: # We use a single mini-batch for training to overfit it.
        lengths, texts, sound_files = lengths[:hp.batch_size]*1000, texts[:hp.batch_size]*1000, sound_files[:hp.batch_size]*1000
    else:
        lengths, texts, sound_files = lengths[:-hp.num_samples], texts[:-hp.num_samples], sound_files[:-hp.num_samples]
    return lengths, texts, sound_files
 
def load_eval_data():
    """We evaluate on the last num_samples."""
    _,texts, _ = create_train_data()
    if hp.sanity_check: # We generate samples for the same texts as the ones we've used for training.
        texts = texts[:hp.batch_size]
    else:
        texts = texts[-hp.num_samples:]
    
    X = np.zeros(shape=[len(texts), hp.max_len], dtype=np.int32)
    for i, text in enumerate(texts):
        _text = np.fromstring(text, np.int32) # byte to int 
        X[i, :len(_text)] = _text
    
    return X
 

