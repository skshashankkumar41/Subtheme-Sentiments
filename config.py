import torch
from transformers import BertTokenizer

INPUT_FILE = "input/Evaluation-dataset.csv"
MAX_LEN = 128
PRE_TRAINED_MODEL = "bert-base-uncased"
NUM_LABELS = 23
MODEL_PATH = "output/bert_model.pth.tar"
ENCODER_PATH = "output/encoder.pkl"
LEARNING_RATE = 1e-04
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)