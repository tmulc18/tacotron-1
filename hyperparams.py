# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

class Hyperparams(object):
    '''Hyper parameters'''
    # mode
    sanity_check = False
    
    # data
    text_file = 'WEB/text.csv'
    sound_fpath = 'WEB'
    max_len = 100 if not sanity_check else 30 # maximum length of text for global training
    min_len = 10 if not sanity_check else 20 # minimum length of text for global training
    
    # signal processing
    sr = 22050 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations 
    use_log_magnitude = True # if False, use magnitude
    
    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
    attention_type = 'luong'  #the type of attention value: luong, bahd
    
    # training scheme
    lr = 0.00015 # Paper => Exponential decay
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

    # Distributed computing
    # There are len(ips) machines, each with n worker nodes and 1 ps on port 2222
    n = 4
    #ips = ['35.197.23.34','35.197.13.2','35.197.33.243','104.196.252.37','35.197.51.142','35.185.201.89']
    ips = ['35.197.42.178','35.185.225.44']
    ps = [ip+':2222' for ip in ips]
    worker = [ip+':'+str(2223+i) for ip in ips for i in range(n)]
    #ps= ['localhost:2222']
    #worker=['localhost:2223','localhost:2224','localhost:2225','localhost:2226']                     
    cluster_spec = {'ps':ps,'worker':worker}
    settle_steps = 0 # use if fewer workers in start of training else set to 0

    # Synchronous
    synch = True
    n_stale = 4
    
    # Binning
    binning = True
    wkr = None # current worker

