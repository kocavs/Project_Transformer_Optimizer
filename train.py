#IMPORT function and class from our own files
from models import BertClassifier
from DataLoad import dataloader

import torch
from torch.distributed.pipeline.sync import Pipe
from torch.distributed import rpc
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids, attention_mask, token_type_ids, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name', type=str, default='bert-base-cased', help='Name of the pre-trained BERT model')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--lr', type=int, default=2e-5, help='Learning Rate')
opts = parser.parse_args()

train_loader, test_loader = dataloader(name="cola", num_train=1000, num_test=1000, batch_size=64)
model = BertClassifier(opts.pretrained_model_name, opts.num_classes)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * opts.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(opts.epochs):
    avg_train_loss = train(model, train_loader, optimizer, scheduler, device)
    print(f"Training epoch {epoch+1}/{opts.epochs} - Loss: {avg_train_loss:.4f}")













