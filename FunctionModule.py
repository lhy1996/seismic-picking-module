import torch
from torch.utils.data import DataLoader, Dataset

def store_model_checkpoint(path, model):
    torch.save({
            'model_state_dict': model.state_dict(),
            }, path)
    return True

def load_model_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


