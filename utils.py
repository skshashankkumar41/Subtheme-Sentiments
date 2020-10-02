import torch 

def save_checkpoint(state, filename="output/bert_model.pth.tar"):
    print("=> Saving Model")
    torch.save(state, filename)

