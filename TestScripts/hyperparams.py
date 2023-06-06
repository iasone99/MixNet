# Audio
num_mels = 64 #80
preemphasis = 0.97

# num_freq = 1024
n_fft = 512 #1024
sr = 16000
hop_length = 160 #200 # samples.
win_length = 400 #800
n_mels = 64 #80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 256
embedding_size = 512
max_db = 100
ref_db = 20

hidden_size_LSTM=500
layers_LSTM=5
    
n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 10000
lr = 0.001
save_step = 10
image_step = 500
n_mfcc=20
layers_DNN=5

gradient_accumulations = 1
weight_decay=0.0
learning_rate_decay_interval = 100  # decay for every 100 epochs
learning_rate_decay_rate = 0.8  # lr = lr * rate

cleaners='english_cleaners'
