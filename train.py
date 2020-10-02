import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertConfig
from model import SentimentMultilabel
from dataloader import get_loader
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = BertConfig()
num_labels = 23
lr = 1e-04
epochs = 10

eval_metrics = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "training_f1_micro": [],
            "training_f1_macro": [],
            "val_f1_micro": [],
            "val_f1_macro": [],
            "training_hamming_loss": [],
            "val_hamming_loss": [],
        }

def loss_fun(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def print_metrics(true, pred, loss, type):
    pred = np.array(pred) >= 0.35
    hamming_loss = metrics.hamming_loss(true,pred)
    precision_micro = metrics.precision_score(true, pred, average='micro',zero_division = 1)
    recall_micro = metrics.recall_score(true, pred, average='micro',zero_division = 1)
    precision_macro = metrics.precision_score(true, pred, average='macro',zero_division = 1)
    recall_macro = metrics.recall_score(true, pred, average='macro',zero_division = 1)
    f1_score_micro = metrics.f1_score(true, pred, average='micro',zero_division = 1)
    f1_score_macro = metrics.f1_score(true, pred, average='macro',zero_division = 1)
    print("-------{} Evaluation--------".format(type))
    print("BCE Loss: {:.4f}".format(loss))
    print("Hamming Loss: {:.4f}".format(hamming_loss))
    print("Precision Micro: {:.4f}, Recall Micro: {:.4f}, F1-measure Micro: {:.4f}".format(precision_micro, recall_micro, f1_score_micro))
    print("Precision Macro: {:.4f}, Recall Macro: {:.4f}, F1-measure Macro: {:.4f}".format(precision_macro, recall_macro, f1_score_macro))
    print("------------------------------------")
    return f1_score_micro, f1_score_macro, hamming_loss, loss 

def save_metrics(eval_metrics,file_name):
    eval = open('output/{}_metrics.pkl'.format(file_name), 'ab') 
    pickle.dump(eval_metrics, eval)                      
    eval.close()
    return True

def train():
    
    model = SentimentMultilabel(num_labels,model_config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    trainLoader, testLoader, _ = get_loader('output/')

    for epoch in range(1,epochs+1):
        eval_metrics["epochs"].append(epoch)
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_f1_micro, train_f1_macro, train_hamming,train_loss = print_metrics(train_targets,train_outputs,epoch_loss, 'Training')
        val_f1_micro, val_f1_macro, val_hamming, val_loss = validate(model, testLoader)
        eval_metrics['training_f1_micro'].append(train_f1_micro)
        eval_metrics['training_f1_macro'].append(train_f1_macro)
        eval_metrics['training_hamming_loss'].append(train_hamming)
        eval_metrics['val_f1_micro'].append(val_f1_micro)
        eval_metrics['val_f1_macro'].append(val_f1_macro)
        eval_metrics['val_hamming_loss'].append(val_hamming)
        eval_metrics["train_loss"].append(train_loss)
        eval_metrics["val_loss"].append(val_loss)
    
    save_metrics(eval_metrics,'bert_base')
    return True

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
            loss = loss_fun(outputs, targets)
            epoch_loss = loss.item()
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        return print_metrics(val_targets,val_outputs, epoch_loss,'Validation')

if __name__ == "__main__":
    train()
    