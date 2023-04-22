#IMPORT function and class from our own files
from DataLoad import dataloader
from transformers import AutoModelForSequenceClassification
import torch
from torch.distributed.pipeline.sync import Pipe
from torch.distributed import rpc
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in train_loader:
        labels = batch['labels'].to(device)
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss

def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            corrected_labels = torch.where(labels == -1, torch.zeros_like(labels), labels)
            output = model(input_ids, attention_mask, token_type_ids)
            logits = output.logits
            loss = torch.nn.CrossEntropyLoss()(logits, corrected_labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == corrected_labels).sum().item()
            total_correct += correct
            total_samples += corrected_labels.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    return average_loss, accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased', help='Name of the pre-trained BERT model')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--lr', type=int, default=5, help='Learning Rate')
opts = parser.parse_args()

train_loader, test_loader = dataloader(name="cola", num_train=1000, num_test=1000, batch_size=64)
model = AutoModelForSequenceClassification.from_pretrained(opts.pretrained_model_name, num_labels=2)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

for epoch in range(opts.epochs):
    avg_train_loss = train(model, train_loader, optimizer, scheduler)
    avg_test_loss, acc = evaluate(model, test_loader)
    print(f"Training epoch {epoch+1}/{opts.epochs} - Training Loss: {avg_train_loss:.4f} - Testing Loss: {avg_test_loss:.4f} - Accuracy: {(acc*100):.4f}%")













