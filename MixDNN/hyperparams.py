# Audio
num_mels = 64  # 80
preemphasis = 0.97

# num_freq = 1024
n_fft = 512  # 1024
sr = 22050
hop_length = 160  # 160 # samples.
win_length = 400  # 400
power = 1.2  # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
embedding_size = 512
max_db = 100
ref_db = 20
n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 10000
lr = 0.1
save_step = 100
image_step = 500
n_mfcc = 20
layers_DNN = 15
hidden_size_DNN = 2048
num_frames = 800
chunk_size = 20
gradient_accumulations = 5
weight_decay = 0.0
learning_rate_decay_interval = 100  # decay for every 100 epochs
learning_rate_decay_rate = 0.999  # lr = lr * rate
device = 'cpu'
cleaners = 'english_cleaners'
