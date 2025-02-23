# Audio parameters (FIXED)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 16
MAX_EPOCHS = 50
ATTENTION_CONTEXT_SIZE = (80, 3)

# Path
PRETRAINED_ENCODER_WEIGHT = './weights/small_encoder.pt'
BG_NOISE_PATH = ["/wp/data/AudioSet", "/wp/data/musan", "/wp/data/FSDnoisy18k"]
TRAIN_MANIFEST = ["/wp/data/dataset_commonvoice_17_vi/manifest_fixed_train_other_validated.jsonl"]
VAL_MANIFEST = ["/wp/data/dataset_commonvoice_17_vi/manifest_fixed_train_other_validated.jsonl"]
LOG_DIR = './checkpoints'

# Optimizer and scheduler parameters
TOTAL_STEPS = 3000000
WARMUP_STEPS = 2000
LR = 1e-4
MIN_LR = 1e-5

# Tokenizer parameters
VOCAB_SIZE = 1024
TOKENIZER_MODEL_PATH = '/wp/tokenizer_spe_bpe_v1024_pad/tokenizer.model'
RNNT_BLANK = 1024
PAD = 1 # tokenizer.pad_id()

# Greedy decoding paramesters
MAX_SYMBOLS = 3

# Whisper-small parameters
N_STATE = 768
N_HEAD = 12
N_LAYER = 12