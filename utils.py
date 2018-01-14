# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
from __future__ import print_function

import copy

import librosa
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf


def get_spectrograms(sound_file): 
    '''Extracts melspectrogram and log magnitude from given `sound_file`.
    Args:
      sound_file: A string. Full path of a sound file.

    Returns:
      Transposed S: A 2d array. A transposed melspectrogram with shape of (T, n_mels)
      Transposed magnitude: A 2d array.Has shape of (T, 1+hp.n_fft//2)
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr) # or set sr to hp.sr.

    # Trimming
    if hp.trim:
        y, _ = librosa.effects.trim(y, top_db=30)

    # Preemphasis
    if hp.preemphasis is not None:
        y = preemphasis(y)
    
    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft, 
                     hop_length=hp.hop_length, 
                     win_length=hp.win_length) 
    
    # magnitude spectrogram
    mag = np.abs(D) #(1+n_fft/2, T)

    # mel spectrogram
    mel = librosa.feature.melspectrogram(S=mag, n_mels=hp.n_mels) #(n_mels, T)

    # Transpose
    mag = mag.astype(np.float32).T
    mel = mel.astype(np.float32).T

    # Transfrom and normalize
    mel,mag = normalize(amp_to_db(mel)),normalize(amp_to_db(mag))

    return mel, mag # (T, n_mels), (T, 1+n_fft/2)
            
def shift_by_one(inputs):
    '''Shifts the content of `inputs` to the right by one 
      so that it becomes the decoder inputs.
      
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
    Returns:
      A 3d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.zeros_like(inputs[:, :1, :]), inputs[:, :-1, :]), 1)

def reduce_frames(arry, step, r):
    '''Reduces and adjust the shape and content of `arry` according to r.
    
    Args:
      arry: A 2d array with shape of [T, C]
      step: An int. Overlapping span.
      r: Reduction factor
     
    Returns:
      A 2d array with shape of [-1, C*r]
    '''
    T = arry.shape[0]
    num_padding = (step*r) - (T % (step*r)) if T % (step*r) !=0 else 0
    
    arry = np.pad(arry, [[0, num_padding], [0, 0]], 'constant', constant_values=(0, 0))
    # T, C = arry.shape
    # sliced = np.split(arry, list(range(step, T, step)), axis=0)
    
    # started = False
    # for i in range(0, len(sliced), r):
    #     if not started:
    #         reshaped = np.hstack(sliced[i:i+r])
    #         started = True
    #     else:
    #         reshaped = np.vstack((reshaped, np.hstack(sliced[i:i+r])))
            
    # return reshaped
    return arry

def spectrogram2wav(spectrogram):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    spectrogram = db_to_amp(denormalize(spectrogram)) ** hp.power
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    if preemphasis is not None:
        y = inv_preemphasis(y)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def restore_shape(arry, step, r):
    '''Reduces and adjust the shape and content of `arry` according to r.
    
    Args:
      arry: A 2d array with shape of [T, C]
      step: An int. Overlapping span.
      r: Reduction factor
     
    Returns:
      A 2d array with shape of [-1, C*r]
    '''
    T, C = arry.shape
    sliced = np.split(arry, list(range(step, T, step)), axis=0)
    
    started = False
    for s in sliced:
        if not started:
            restored = np.vstack(np.split(s, r, axis=1))
            started = True
        else:    
            restored = np.vstack((restored, np.vstack(np.split(s, r, axis=1))))
    
    # Trim zero paddings
    restored = restored[:np.count_nonzero(restored.sum(axis=1))]    
    return restored


def preemphasis(x):
    """Applies preemphasis."""
    return signal.lfilter([1, -hp.preemphasis], [1], x)


def inv_preemphasis(x):
    """Inverse of preemphasis."""
    return signal.lfilter([1], [1, -hp.preemphasis], x)


def amp_to_db(x):
    """Converts from amplitutde to decibel."""
    return 20 * np.log10(np.maximum(1e-5, x))- hp.ref_level_db


def db_to_amp(x):
    """Converts from decibel to amplitude."""
    return np.power(10.0, (x + hp.ref_level_db)* 0.05)


def normalize(S):
    """Normalized tensor to be in [0,1]."""
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def denormalize(S):
    """Inverse of normalization."""
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db


def plot_alignment(alignment,gs):
    '''Plots the alignment.

    alignment: (numpy) matrix of shape (encoder_steps,decoder_steps)
    gs : (int) global step
    '''
    fig, ax = plt.subplots()
    im=ax.imshow(alignment,aspect='auto',interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.xlabel('Decoder timestep'+ '\n\n' +'step: '+ str(gs))
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(hp.logdir+'/alignment_%d'%gs,format='png')
