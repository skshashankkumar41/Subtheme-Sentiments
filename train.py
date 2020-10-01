import torch
from transformers import BertTokenizer, BertConfig
from model import SentimentMultilabel
from dataloader import get_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = BertConfig()
num_labels = 27
lr = 1e-05
epochs = 10

metrics = {
            "epoch_loss": [],
            "step_loss": [],
            "f1_score_macro": [],
            "f1_score_micro": [],
            "accuracy": []
        }

# ref- https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLossLogits(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLossLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.BCEWithLogitsLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)

        f_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return torch.mean(f_loss)

def loss_fn(outputs, targets):
    return FocalLossLogits()(outputs, targets)

def train():
    model = SentimentMultilabel(num_labels,model_config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    trainLoader, testLoader, _ = get_loader('output/')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for _, data in enumerate(trainLoader):
            ids = data['ids'].to(self.device, dtype=torch.long)
            mask = data['mask'].to(self.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fun(outputs, targets)
            epoch_loss = loss.item()
            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                metrics["step_loss"].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #self.validate(testing_loader)

        metrics["epoch_loss"].append(epoch_loss)

    

    