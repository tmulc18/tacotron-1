# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

class Hyperparams:
    '''Hyper parameters'''
    # mode
    sanity_check = False
    
    # data
    text_file = 'Kate/text.tsv'
    sound_fpath = 'Kate'
    max_len = 180 if not sanity_check else 30 # maximum length of text
    min_len = 10 if not sanity_check else 20 # minimum length of text
    LJ = True
    
    # signal processing
    sr = 22050 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 30 # Number of inversion iterations
    preemphasis = .97 # or None
    min_level_db = -100
    ref_level_db = 20
    trim = False

    
    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    norm_type = 'bn'  # a normalizer function. value: bn, ln, ins, or None
    
    # training scheme
    lr = 0.001 # Paper => Exponential decay
    logdir = "logdir" if not sanity_check else "logdir_s"
    outputdir = 'samples' if not sanity_check else "samples_s"
    batch_size = 32
    num_epochs = 10000 if not sanity_check else 40 # Paper => 2M global steps!
    loss_type = "l1" # Or you can test "l2"
    num_samples = 32
    
    # etc
    num_gpus = 1 # If you have multiple gpus, adjust this option, and increase the batch size
                 # and run `train_multiple_gpus.py` instead of `train.py`.
    target_zeros_masking = False # If True, we mask zero padding on the target, 
                                 # so exclude them from the loss calculation.     
