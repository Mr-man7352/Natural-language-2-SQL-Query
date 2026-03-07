# ── Model & tokenizer ─────────────────────────────────────────────────────────
MODEL_NAME = "t5-small"
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 128

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
NUM_EPOCHS = 2
SAVE_STEPS = 50
LOGGING_STEPS = 50
MAX_TRAIN_SAMPLES = 8000

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_OUTPUT_DIR = "./sql_model"
DATASET_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
ARCHIVE_NAME = "data.tar.bz2"

# ── Inference ─────────────────────────────────────────────────────────────────
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
