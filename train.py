import torch
import numpy as np
from transformers import BertTokenizer, BertConfig
from model import SentimentMultilabel
from dataloader import get_loader
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = BertConfig()
num_labels = 23
lr = 1e-05
epochs = 10

eval_metrics = {
            "epoch_loss": [],
            "step_loss": [],
            "f1_score_macro": [],
            "f1_score_micro": [],
            "accuracy": []
        }

# ref- https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# class FocalLossLogits(torch.nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLossLogits, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = torch.nn.BCEWithLogitsLoss()(inputs, targets)

#         pt = torch.exp(-BCE_loss)

#         f_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

#         return torch.mean(f_loss)

# def loss_fun(outputs, targets):
#     return FocalLossLogits()(outputs, targets)

def loss_fun(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def print_metrics(true, pred, type):
    pred = np.array(pred) >= 0.5
    hamming_loss = metrics.hamming_loss(true,pred)
    precision_micro = metrics.precision_score(true, pred, average='micro')
    recall_micro = metrics.recall_score(true, pred, average='micro')
    precision_macro = metrics.precision_score(true, pred, average='macro')
    recall_macro = metrics.recall_score(true, pred, average='macro')
    f1_score_micro = metrics.f1_score(true, pred, average='micro')
    f1_score_macro = metrics.f1_score(true, pred, average='macro')
    print("-------{} Evaluation--------")
    print("Hamming Loss: {:.4f}".format(hamming_loss))
    print("Precision Micro: {:.4f}, Recall Micro: {:.4f}, F1-measure Micro: {:.4f}".format(precision_micro, recall_micro, f1_score_micro))
    print("Precision Macro: {:.4f}, Recall Macro: {:.4f}, F1-measure Macro: {:.4f}".format(precision_macro, recall_macro, f1_score_macro))
    print("------------------------------------")

def train():
    
    model = SentimentMultilabel(num_labels,model_config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    trainLoader, testLoader, _ = get_loader('output/')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_targets = []
        train_outputs = []
        for _, data in enumerate(trainLoader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fun(outputs, targets)
            epoch_loss = loss.item()
            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                eval_metrics["step_loss"].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print_metrics(train_targets,train_outputs,'Training')
        validate(model, testLoader)

        eval_metrics["epoch_loss"].append(epoch_loss)

def validate(model, testLoader):
    model.eval()
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testLoader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        print_metrics(val_targets,val_outputs,'Validation')
        return True


if __name__ == "__main__":
    train()
    


    

    