from PIL import Image
from tqdm import tqdm_notebook as tqdm
import torch
import pandas as pd
import torch.nn as nn
from preprocess.image_transform import ImageTransform
import torch.nn.functional as F

def make_test_result_dataframe(test_images_filepaths: list,size: int,mean: tuple,std: tuple,model:nn.Module,device:str)-> pd.DataFrame:
    id_list = []
    pred_list = []
    _id = 0
    with torch.no_grad():
        for test_path in tqdm(test_images_filepaths):
            img = Image.open(test_path)
            _id = test_path.split("/")[-1].split(".")[1]
            transform = ImageTransform(size, mean, std)
            img = transform(img, phase="val")
            img = img.unsqueeze(0)
            img = img.to(device)

            model.eval()
            outputs = model(img)
            preds = F.softmax(outputs, dim=1)[:, 1].tolist()

            id_list.append(_id)
            pred_list.append(preds[0])

    res = pd.DataFrame({"id": id_list, "label": pred_list})
    return res

def calculate_accuracy(y_pred: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)