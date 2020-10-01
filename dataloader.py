from dataset import SentimentDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

MAX_LEN = 64
PRE_TRAINED_MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

def get_loader(rootPath, train_batch_size=8, test_batch_size = 4, shuffle=True, num_workers=8, pin_memory=True):
    trainDataset = SentimentDataset(rootPath + 'train.pkl', tokenizer, MAX_LEN)
    testDataset = SentimentDataset(rootPath + 'test.pkl', tokenizer, MAX_LEN)

    trainLoader = DataLoader(
        dataset=trainDataset,
        batch_size=train_batch_size,
        shuffl=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    testLoader = DataLoader(
        dataset=testDataset,
        batch_size=test_batch_size,
        shuffl=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return trainLoader, testLoader, tokenizer
