import time
import os
import tensorflow as tf
import numpy as np
params_carlini = False
time_suffix = time.strftime('%H%M%S_%Y%m%d')
save_path_prefix = 'summary_model/exp_{}'.format(time_suffix)
os.makedirs(save_path_prefix, exist_ok=True)
def debug(string):
    # if hparams.debug:
    print(string)

in_type = "mulaw-quantize"


out_channels = 3*10 if in_type!='mulaw-quantize' else 256
in_channels = 1 if in_type!='mulaw-quantize' else out_channels
# Default hyperparameters
hparams = tf.contrib.training.HParams(

#################### new params #############################
    gpu_ids=['0'],
    isTrain = True,
    preprocess = 'scale_width',
    continue_train=False,
    load_iter=0,
    verbose=False,
    lr_policy='linear', #learning rate policy. [linear | step | plateau | cosine]
    audio_max_length = 110000,#here max is 88320
    audio_preprocess=True, #use when load dataset
    input_nc = 512,
    output_nc = 512,
    ngf = 64,
    ndf=64,
    netG='unet_128',
    norm='batch',
    no_dropout=True,
    init_type='normal',
    init_gain=0.02,
    netD='basic',
    n_layers_D=3,
    pool_size =3,
    gan_mode='lsgan',#define the gan loss
    lr=2e-4,
    beta1=0.5,
    lambda_identity=0.5,
    lambda_A=1.0,
    niter=100, #which epoch we should change the starting lr
    niter_decay=100, #how many epoch it takes to the lowest lr
    lr_decay_iters=50,
    print_freq=10,
    direction="AtoB",


#################### new params #############################
    summary_path = save_path_prefix,


    params_path = save_path_prefix,
    samples_path = os.path.join(save_path_prefix,'samples'),
    restore_dir=None,#r"/usr/whz/generative_audio_attack/gan_model/summary_model/exp_172815_20190112",
    debug=True,
    phase='train',
    pretrain_wavenet=False,
    n_epochs=100,
    save_freq=1,
    epoch_iters=100,
    init_lr=1e-4,
    lambda_update_rate=0.5,
    end_lr = 5e-6,
    wave_path=os.path.join('..', 'waves'),
    wave_prep_path=os.path.join('..', 'preprocessed_waves'),
    mel_path=os.path.join('..', 'melspec'),
    batch_size=1,
    eval_batch_size=1,
    record_path=os.path.join(r'..','tfrecords'),
    logger_name = None, # TODO
    use_speaker_embedding=False,
    sample_rate=16000,  # 22050 Hz (corresponding to ljspeech dataset)
	##########################################################################################################
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners".
    cleaners='english_cleaners',

    # Hardware setup
    use_all_gpus=True,
    # Whether to use all GPU resources. If True, total number of available gpus will override num_gpus.
    num_gpus=3,  # Determines the number of gpus in use
    lr_decay = 0.9,
    ###########################################################################################################################################

    # Audio
    # because deepspeech has 26 mel channels
    num_mels=26 if params_carlini else 80,  # 80, Number of mel-spectrogram channels and local conditioning dimensionality
    cin_channels=26 if params_carlini else 80,  # 80, Set this to -1 to disable local conditioning, else it must be equal to num_mels!!
    num_freq=513,  # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    rescale=False if params_carlini else True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.999,  # Rescaling value
    trim_silence=False if params_carlini else True,  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    clip_mels_length=True,  # For cases of OOM (Not really recommended, working on a workaround)
    max_mel_frames=900,  # Only relevant when clip_mels_length = True

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False if params_carlini else True, # modified
    silence_threshold=2,  # silence threshold used for sound trimming for wavenet preprocessing

    # Mel spectrogram
    n_fft=512 if params_carlini else 1024,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=160 if params_carlini else 256,  # For 22050Hz, 275 ~= 12.5 ms
    win_size=400 if params_carlini else None,  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
    frame_shift_ms=None,

    # M-AILABS (and other datasets) trim params
    trim_fft_size=512,
    trim_hop_size=128,
    trim_top_db=60,

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=False if params_carlini else True, # modified
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,  # Whether to scale the data to be symmetric around 0
    max_abs_value=4.,  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]

    # Global style token
    use_gst=True,
    # When false, the scripit will do as the paper  "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
    num_gst=10,
    num_heads=4,  # Head number for multi-head attention
    style_embed_depth=256,
    audio_enc_filters=[32, 32, 64, 64, 128, 128],
    audio_enc_units=128,
    style_att_type="mlp_attention",  # Attention type for style attention module (dot_attention, mlp_attention)
    style_att_dim=128,

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=25,
    # Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
    fmax=7600,

    # Griffin Lim
    power=1.2,
    griffin_lim_iters=60,
    ###########################################################################################################################################

    # Tacotron
    outputs_per_step=1 if params_carlini else 2, # TODO outputs per step,
    # the decoder outputs is the multiple of this, and the target requires padding, otherwise causes dimension mismatch
    # number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
    stop_at_any=True,
    # Determines whether the decoder should stop when predicting <stop> to any frame or to all of them

    embedding_dim=256 if params_carlini else 256,  # dimension of embedding space

    enc_conv_num_layers=3,  # number of encoder convolutional layers
    enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
    enc_conv_channels=128,  # number of encoder convolutions filters for each layer
    encoder_lstm_units=64,  # number of lstm units for each direction (forward and backward)

    smoothing=False,  # Whether to smooth the attention normalization function
    attention_dim=128,  # dimension of attention space
    attention_filters=32,  # number of attention convolution filters
    attention_kernel=(31,),  # kernel size of attention convolution
    cumulative_weights=True,
    # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

    prenet_layers=[256, 256],  # number of layers and number of units of prenet
    decoder_layers=2,  # number of decoder lstm layers
    decoder_lstm_units=1024,  # number of decoder lstm units on each layer
    max_iters=2500,  # Max decoder steps during inference (Just for safety from infinite loop cases)

    postnet_num_layers=5,  # number of postnet convolutional layers
    postnet_kernel_size=(5,),  # size of postnet convolution filters for each layer
    postnet_channels=512,  # number of postnet convolution filters for each layer

    mask_encoder=True,  # whether to mask encoder padding while computing attention
    mask_decoder=True,
    # Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)

    cross_entropy_pos_weight=20,
    # Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
    predict_linear=False,
    # Whether to add a post-processing network to the Tacotron to predict linear spectrograms (True mode Not tested!!)
    ###########################################################################################################################################

    # Wavenet
    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot
    # input and softmax output are assumed.
    input_type='raw' if params_carlini else in_type,
    in_channels = in_channels,
    quantize_channels=256,
    # 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255

    log_scale_min=float(np.log(1e-14)),  # Mixture of logistic distributions minimal log scale

    out_channels=out_channels,
    # This should be equal to quantize channels when input type is 'mulaw-quantize' else: num_distributions * 3 (prob, mean, log_scale)
    layers=12,#24,  # Number of dilated convolutions (Default: Simplified Wavenet of Tacotron-2 paper)
    stacks=4,  # Number of dilated convolution stacks (Default: Simplified Wavenet of Tacotron-2 paper)
    residual_channels=256,#512,
    gate_channels=256,#512,  # split in 2 in gated convolutions
    skip_out_channels=256,#256,
    kernel_size=3,


    upsample_conditional_features=True,
    # Whether to repeat conditional features or upsample them (The latter is recommended)
    upsample_scales=[16, 16],  # prod(scales) should be equal to hop size
    freq_axis_kernel_size=3,

    gin_channels=-1 if params_carlini else 128,  # Set this to -1 to disable global conditioning, Only used for multi speaker dataset
    use_bias=True,  # Whether to use bias in convolutional layers of the Wavenet

    max_time_sec=None,
    max_time_steps=13000,  # Max time steps in audio used to train wavenet (decrease to save memory)
    ###########################################################################################################################################

    # Tacotron Training
    tacotron_random_seed=5339,
    # Determines initial graph and operations (i.e: model) random state for reproducibility
    tacotron_swap_with_cpu=False,
    # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

    tacotron_batch_size=48,  # number of training samples on each training steps
    tacotron_reg_weight=1e-6,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=True,
    # Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

    tacotron_test_size=None,  # % of data to keep as test data, if None, tacotron_test_batches must be not None
    tacotron_test_batches=48,  # number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
    tacotron_data_random_state=1234,  # random state for train test split repeatability

    tacotron_decay_learning_rate=True,  # boolean, determines if the learning rate will follow an exponential decay
    tacotron_start_decay=50000,  # Step at which learning decay starts
    tacotron_decay_steps=40000,  # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate=0.2,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-5,  # minimal learning rate

    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer beta3 parameter

    tacotron_zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
    tacotron_dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet

    natural_eval=False,
    # Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

    # Decoder RNN learning can take be done in one of two ways:
    #	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
    #	Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
    # The second approach is inspired by:
    # Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
    # Can be found under: https://arxiv.org/pdf/1506.03099.pdf
    tacotron_teacher_forcing_mode='constant',
    # Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
    tacotron_teacher_forcing_ratio=1.,
    # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
    tacotron_teacher_forcing_init_ratio=1.,  # initial teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_final_ratio=0.,  # final teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_start_decay=10000,
    # starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_steps=280000,
    # Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_alpha=0.,  # teacher forcing ratio decay rate. Relevant if mode='scheduled'
    ###########################################################################################################################################

    # Wavenet Training
    wavenet_random_seed=5339,  # S=5, E=3, D=9 :)
    wavenet_swap_with_cpu=False,
    # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

    wavenet_batch_size=4,  # batch size used to train wavenet.
    wavenet_test_size=0.0441,  # % of data to keep as test data, if None, wavenet_test_batches must be not None
    wavenet_test_batches=None,  # number of test batches.
    wavenet_data_random_state=1234,  # random state for train test split repeatability

    wavenet_learning_rate=1e-4,
    wavenet_adam_beta1=0.9,
    wavenet_adam_beta2=0.999,
    wavenet_adam_epsilon=1e-6,

    wavenet_ema_decay=0.9999,  # decay rate of exponential moving average

    wavenet_dropout=0.05,  # drop rate of wavenet layers
    train_with_GTA=False,  # Whether to use GTA mels to train WaveNet instead of ground truth mels.
    ###########################################################################################################################################

    # Eval sentences (if no eval file was specified, these sentences are used for eval)
    sentences=[
        # From July 8, 2017 New York Times:
        'Scientists at the CERN laboratory say they have discovered a new particle.',
        'There\'s a way to measure the acute emotional intelligence that has never gone out of style.',
        'President Trump met with other leaders at the Group of 20 conference.',
        'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
        # From Google's Tacotron example page:
        'Generative adversarial network or variational auto-encoder.',
        'Basilar membrane and otolaryngology are not auto-correlations.',
        'He has read the whole thing.',
        'He reads books.',
        "Don't desert me here in the desert!",
        'He thought it was time to present the present.',
        'Thisss isrealy awhsome.',
        'Punctuation sensitivity, is working.',
        'Punctuation sensitivity is working.',
        "The buses aren't the problem, they actually provide a solution.",
        "The buses aren't the PROBLEM, they actually provide a SOLUTION.",
        "The quick brown fox jumps over the lazy dog.",
        "does the quick brown fox jump over the lazy dog?",
        "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
        "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
        "The blue lagoon is a nineteen eighty American romance adventure film.",
        "Tajima Airport serves Toyooka.",
        'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
        # From Training data:
        'the rest being provided with barrack beds, and in dimensions varying from thirty feet by fifteen to fifteen feet by ten.',
        'in giltspur street compter, where he was first lodged.',
        'a man named burnett came with his wife and took up his residence at whitchurch, hampshire, at no great distance from laverstock,',
        'it appears that oswald had only one caller in response to all of his fpcc activities,',
        'he relied on the absence of the strychnia.',
        'scoggins thought it was lighter.',
        '''would, it is probable, have eventually overcome the reluctance of some of the prisoners at least, 
        and would have possessed so much moral dignity''',
        '''Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization. 
        This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that 
        the adopted architecture is able to perform this task with wild success.''',
        'Thank you so much for your support!',
    ]

)

