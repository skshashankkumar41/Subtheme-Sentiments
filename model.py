import torch
from transformers import BertModel, BertPreTrainedModel, BertTokenizer

PRE_TRAINED_MODEL = "bert-base-uncased"

# Bert Pretrained model with final classifier 
class SentimentMultilabel(BertPreTrainedModel):
    def __init__(self, num_labels, config):
        super(SentimentMultilabel, self).__init__(config)
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)
        self.drop = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.drop(pooled_output)
        output = self.classifier(output)
        return output