from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np


class BaseDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y
  def __len__(self):
    return len(self.x)
  def __getitem__(self, idx):
    text = self.x.iloc[idx]
    label = self.y.iloc[idx]
    return text, label

def load_model():
  pipe = pipeline("text-classification", model="snunlp/KR-FinBert-SC")
  tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
  model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")
  model.classifier = torch.nn.Linear(768, 2)

  for para in model.parameters():
    para.requires_grad=False
  for name, param in model.named_parameters():
    if "classifier" in name:
      param.requires_grad=True
  return pipe, tokenizer, model

def load_dataset(path='hawkishdorvishlabel.csv'):
  df_label = pd.read_excel(path)
  return df_label

def load_dataloader(df_label,batch=1024):
  x_train, x_test, y_train, y_test = train_test_split(df_label['본문'], df_label['label'], test_size = 0.3, random_state = 123, stratify = df_label['label'])
  test_dataset = BaseDataset(x_test, y_test)
  train_dataset = BaseDataset(x_train, y_train)
  test_dataloader = DataLoader(test_dataset, batch_size = batch, shuffle = True)
  train_dataloader = DataLoader(train_dataset, batch_size = batch, shuffle = True)
  return test_dataset, train_dataset, test_dataloader, train_dataloader

def train(tokenizer,model, train_dataloader, test_dataloader,model_save_path='/content/drive/MyDrive/mdl.pt'):
  total_steps=len(train_dataloader)*1000
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=1e-3)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  train_history=[] #loss
  test_history=[] #accuarcy
  current_acc = 0
  for epoch in tqdm(range(1000)):
    # 모델 계산
    model.train()
    loss_item = 0
    for input, labels in train_dataloader:
      encoded_input=tokenizer(input,padding=True,truncation=True,max_length=512,return_tensors='pt')
      encoded_input=encoded_input.to(device)
      labels=torch.tensor(labels).to(device)
      optimizer.zero_grad()
      outputs = model(**encoded_input)
      loss=criterion(outputs.logits, labels)
      loss_item+=loss.item()
      loss.backward()
      optimizer.step()
      scheduler.step()
    avg_train_loss=loss_item/len(input)
    if epoch % 50 == 0:
      print(f'Epoch [{epoch+1}/{500}], Loss: {avg_train_loss:.4f}')
      train_history.append(avg_train_loss)
      model.eval()
      acc_item=0
      with torch.no_grad():
        for input, labels in test_dataloader:
          encoded_input=tokenizer(input,padding=True,truncation=True,max_length=512,return_tensors='pt')
          encoded_input=encoded_input.to(device)
          labels=torch.tensor(labels).to(device)
          outputs = model(**encoded_input)
          result = torch.nn.Softmax(dim=0)(outputs.logits)
          output = torch.topk(result,1).indices
          output=output.flatten()
          acc_item+= torch.sum(output==labels).item()
      acc=acc_item/len(test_dataloader)
      test_history.append(acc)
      print(f'Epoch [{epoch+1}/{500}], Accuracy: {acc:.4f}')
      if current_acc < acc:
        current_acc = acc
        torch.save(model.state_dict(), model_save_path)
  return train_history, test_history

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  _,tokenizer, model = load_model()
  df_label = load_dataset()
  test_dataset, train_dataset, test_dataloader, train_dataloader = load_dataloader(df_label)
  train_history, test_history = train(tokenizer,model, train_dataloader, test_dataloader)
  plt.plot(range(len(train_history)),train_history)
  plt.plot(range(len(test_history)),test_history)
  plt.legend(['train_loss','test_acc'])
  plt.show()
