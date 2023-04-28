#IMPORT function and class from our own files
from DataLoad import dataloader
import torch
import argparse
import time

from transformers import  AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_dataset

from torch.distributed.pipeline.sync import Pipe
from torch.distributed import rpc



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(examples, tokenizer, max_length=512):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": torch.tensor(examples["label"]),
    }

def calcuate_accuracy(preds, labels):
  idx_max = torch.argmax(preds, dim=-1)
  n_correct = (idx_max==labels).sum().item()
  return n_correct

def train(model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    num_correct = 0
    num_total = 0
    for batch in train_loader:
        #print(batch['input_ids'])
        labels = batch['labels'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask).logits
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        num_correct += calcuate_accuracy(outputs, labels)
        total_loss += loss.item()
        num_total += labels.size(0)
        
    avg_train_loss = total_loss / num_total
    avg_train_acc = num_correct / num_total * 100.0
    
    return avg_train_loss, avg_train_acc

def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_total = 0
    with torch.no_grad():
        for batch in test_loader:
            
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids, attention_mask).logits
            loss = torch.nn.CrossEntropyLoss()(output, labels)

            total_loss += loss.item()
            total_correct += calcuate_accuracy(output, labels)
            total_samples += labels.size(0)
            num_total += labels.size(0)

    average_loss = total_loss / num_total
    accuracy = total_correct / num_total * 100.0

    return average_loss, accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased', help='Name of the pre-trained BERT model')
parser.add_argument('--epoch', type=int, default=3, help='Number of training epoches')
parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
parser.add_argument('--lr', type=int, default=5e-5, help='Learning Rate')
opts = parser.parse_args()

train_loader, test_loader = dataloader(name="ag_news", token_name=opts.pretrained_model_name, train_length=5000, test_length=1000, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained(opts.pretrained_model_name, num_labels=opts.num_classes)
model.to(device)

# Set the optimizer
optimizer = AdamW(model.parameters(), lr=opts.lr)

num_training_steps = opts.epoch * len(train_loader)
scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_training_steps
)

for epoch in range(opts.epoch):
    start_time = time.time()
    
    avg_train_loss, avg_train_acc = train(model, train_loader, optimizer, scheduler)
    
    end_time = time.time()
    avg_test_loss, avg_test_acc = evaluate(model, test_loader)
    
    epoch_time = end_time - start_time
    print("Epoch: ", epoch)
    print(f'\tTrain Loss: {avg_train_loss:.5f} | Train Acc: {avg_train_acc:.2f}%')
    print(f'\tTest. Loss: {avg_test_loss:.5f} |  Test Acc: {avg_test_acc:.2f}%')
    print(f"\tTime: {epoch_time:.2f} seconds")










