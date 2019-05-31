# training
GPU_WHEN_TRAINING = "3"
BATCH_SIZE = 128
EPOCH = 8

# setting
MISSION = "accu"  # "accu" or "law"
DATA_PATH = "data_old"

# model
MODEL = "LSTM"  # "GRU" or "LSTM"

LAST_LAYER = "softmax"  # "sigmoid" or "softmax"
SIGMOID_THRESHOLD = -0.84
SOFTMAX_THRESHHOLD = 0.88

MODEL_NAME = "LSTM_"+MISSION+"_"+LAST_LAYER+"_old_data_0706"

