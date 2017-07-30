# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
from __future__ import print_function

import copy

import librosa

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
    
    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft, 
                     hop_length=hp.hop_length, 
                     win_length=hp.win_length) 
    
    # magnitude spectrogram
    magnitude = np.abs(D) #(1+n_fft/2, T)
    
    # power spectrogram
    power = magnitude**2 #(1+n_fft/2, T) 
    
    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels) #(n_mels, T)

    return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (T, n_mels), (T, 1+n_fft/2)
            
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
    T, C = arry.shape
    sliced = np.split(arry, list(range(step, T, step)), axis=0)
    
    started = False
    for i in range(0, len(sliced), r):
        if not started:
            reshaped = np.hstack(sliced[i:i+r])
            started = True
        else:
            reshaped = np.vstack((reshaped, np.hstack(sliced[i:i+r])))
            
    return reshaped

def spectrogram2wav(spectrogram):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)

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

def byte_size_load_fn(op):
    """Load function that computes the byte size of a single-output `Operation`.
    This is intended to be used with `"Variable"` ops, which have a single
    `Tensor` output with the contents of the variable.  However, it can also be
    used for calculating the size of any op that has a single output.
    Intended to be used with `GreedyLoadBalancingStrategy`.
    Args:
      op: An `Operation` with a single output, typically a "Variable" op.
    Returns:
      The number of bytes in the output `Tensor`.
    Raises:
      ValueError: if `op` does not have a single output, or if the shape of the
        single output is not fully-defined.
    """
    if len(op.outputs) != 1:
      raise ValueError("Op %s must have a single output" % op)
    output = op.outputs[0]
    elem_size = output.dtype.size
    shape = output.get_shape()
    if not shape.is_fully_defined():
      # Due to legacy behavior, scalar "Variable" ops have output Tensors that
      # have unknown shape when the op is created (and hence passed to this
      # load function for placement), even though the scalar shape is set
      # explicitly immediately afterward.
      shape = tf.tensor_shape.TensorShape(op.get_attr("shape"))
    shape.assert_is_fully_defined()
    return shape.num_elements() * elem_size

def get_cdf(text_len):
    """
    Given array of text lens, get cdf and its support
    """
    hist,bin_edges=np.histogram(np.array(text_len),bins=50,density=True)
    dx=bin_edges[1] - bin_edges[0]
    cdf=np.cumsum(hist)*dx
    cdf=np.insert(cdf,0,0)
    X = bin_edges
    return cdf,X

def cdf_inv(cdf,X,y):
    """
    Given cdf as array, the support X,  and y = cdf(x), find inverse
    """
    return X[np.argmax(cdf >= y)]

def get_bins(num_workers,text_len):
    """
    Given a list of text lens and the number of workers, compute bins for each worker
    """
    cdf,X = get_cdf(text_len)
    
    percentage_data = 1./num_workers
    if hp.sanity_check:
        percentage_data = 1.
        
    bins = dict()
    for worker in range(num_workers):
        min_q,max_q = (worker)*percentage_data,(worker+1)*percentage_data
        bin_i = (cdf_inv(cdf,X,min_q),cdf_inv(cdf,X,max_q))
        bins[worker]=bin_i
    return bins
