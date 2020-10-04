import torch
from transformers import BertTokenizer

model = "base"
INPUT_FILE = "input/Evaluation-dataset.csv"
MAX_LEN = 128
PRE_TRAINED_MODEL = "bert-base-uncased" if model =='base' else "bert-large-uncased"
NUM_LABELS = 23
MODEL_PATH = "output/bert_model.pth.tar"
ENCODER_PATH = "output/encoder.pkl"
LEARNING_RATE = 1e-04 if model == "base" else 1e-05
EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)