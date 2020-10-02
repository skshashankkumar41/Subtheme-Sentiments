import torch 
import pickle
import numpy as np
from model import SentimentMultilabel
from transformers import BertConfig,BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = BertConfig()
num_labels = 23
MAX_LEN = 128
PRE_TRAINED_MODEL = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
model = SentimentMultilabel(num_labels,model_config).to(device) 
checkpoint = torch.load("output/bert_model.pth.tar")
model.load_state_dict(checkpoint["state_dict"])
encoder = open('output/encoder.pkl', 'rb')      
le = pickle.load(encoder) 
encoder.close() 

def inference(text,model,tokenizer):
    model.eval()
    inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device, dtype=torch.long)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device, dtype=torch.long)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(device, dtype=torch.long)

    outputs = model(ids, mask, token_type_ids).sigmoid()
    
    prediction = [1 if i > 0.35 else 0 for i in outputs[0]]

    labels = le.inverse_transform(np.array([prediction]))[0]
    print("Labels -- {}".format(labels))
    return list(labels)


if __name__ == '__main__':
    labels = inference('Good price for well rated tyres',model,tokenizer)
