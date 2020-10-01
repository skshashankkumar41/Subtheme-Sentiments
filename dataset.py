import torch
from torch.utils.data import Dataset, DataLoader

# Pytorch dataset for the training of BERT model
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.text = self.df.text 
        self.targets = self.df.encoded
        self.max_len = max_len 
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,index):
        text = self.text[index]
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.data['encoded'][index], dtype=torch.long)
        }
