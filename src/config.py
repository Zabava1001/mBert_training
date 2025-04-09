import torch
import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MBERT_PATH = os.path.join(PROJECT_DIR, "/mbart_encoder")
MARIAN_PATH = os.path.join(PROJECT_DIR, "/marian_decoder")


SAVE_PATH = os.path.join(PROJECT_DIR, 'mbert-marian-ru-ha')
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'dataset.xlsx')
OUTPUT_PATH = os.path.join(PROJECT_DIR, 'output-ru-ha')

SAMPLE_SIZE = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры предобработки
MAX_LENGTH = 128  # Максимальная длина последовательности
TEST_SIZE = 0.1  # Доля данных для теста

# Гиперпараметры обучения
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 8
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01

# Опции логирования и сохранения
SAVE_STRATEGY = "epoch"  # Сохранение модели раз в эпоху
SAVE_TOTAL_LIMIT = 2  # Количество сохраняемых чекпоинтов
LOGGING_STEPS = 500  # Как часто логировать метрики

# Аппаратное ускорение
USE_FP16 = torch.cuda.is_available()

TRAINING_ARGS = {
    "output_dir": OUTPUT_PATH,
    "eval_strategy": "epoch",
    "learning_rate": LEARNING_RATE,
    "per_device_train_batch_size": BATCH_SIZE_TRAIN,
    "per_device_eval_batch_size": BATCH_SIZE_EVAL,
    "num_train_epochs": NUM_EPOCHS,
    "weight_decay": WEIGHT_DECAY,
    "save_strategy": SAVE_STRATEGY,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "logging_dir": "./logs",
    "logging_steps": LOGGING_STEPS,
    "fp16": USE_FP16
}
