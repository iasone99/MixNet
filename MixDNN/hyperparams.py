# Audio
num_mels = 64  # 80
warmup_steps = 25

# num_freq = 1024
n_fft = 512  # 1024
sr = 22050
hop_length = 160  # 160 # samples.
win_length = 400  # 400
embedding_size = 512
n_iter = 60

epochs = 10000
lr = 10
save_step = 100
image_step = 500
n_mfcc = 20
layers_DNN = 15
hidden_size_DNN = 2048
num_frames = 800
chunk_size = 20
gradient_accumulations = 1  # 5
weight_decay = 0.0
learning_rate_decay_interval = 50  # decay for every 100 epochs
learning_rate_decay_rate = 0.8  # lr = lr * rate #0.999
device = 'cpu'
cleaners = 'english_cleaners'
