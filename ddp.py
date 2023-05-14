# IMPORT function and class from our own files
import torch
import argparse
import time
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calcuate_accuracy(preds, labels):
    # Calculate the index of the predicted class with the highest probability
    idx_max = torch.argmax(preds, dim=-1)
    # Count the number of correct predictions
    n_correct = (idx_max == labels).sum().item()
    # Return correct number
    return n_correct


def dataloader(name, token_name, train_length, test_length, batch_size, ddp=False):
    # Function to padding and change the tokens according to the pretrained model
    def tokenize_function(examples):
        tokenizer = AutoTokenizer.from_pretrained(token_name)
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    # Download dataset
    datasets = load_dataset(name, cache_dir="./dataset")
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # Select corresponding datasets
    train_datasets = tokenized_datasets["train"].select(range(train_length))
    test_datasets = tokenized_datasets["test"].select(range(test_length))
    # Here we shuffle our train dataloader
    # If ddp, we need to create sampler
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_datasets)
        train_dataloader = DataLoader(train_datasets, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(test_datasets, batch_size=batch_size, sampler=test_sampler)
    else:
        train_sampler = 0
        test_sampler = 0
        train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(test_datasets, batch_size=batch_size)
    return train_dataloader, test_dataloader, train_sampler, test_sampler


def train(model, train_loader, optimizer, scheduler, loss_func, epoch, rank=None):
    model.train()
    total_loss = 0
    num_correct = 0
    num_total = 0
    train_loader.sampler.set_epoch(epoch)
    print("Start Training")
    for batch in train_loader:
        # From CPU to GPU
        labels = batch['labels'].to(rank)
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        # Get output and calculate loss
        outputs = model(input_ids, attention_mask).logits
        loss = loss_func(outputs, labels)
        # Update parameters
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # Calculate loss and acc
        num_correct += calcuate_accuracy(outputs, labels)
        total_loss += loss.item()
        num_total += labels.size(0)
    avg_train_loss = total_loss / num_total
    avg_train_acc = num_correct / num_total * 100.0
    return avg_train_loss, avg_train_acc


def evaluate(model, test_loader, loss_func, epoch, rank=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_total = 0
    test_loader.sampler.set_epoch(epoch)
    # Start evaluation
    with torch.no_grad():
        for batch in test_loader:
            # From CPU to GPU
            labels = batch['labels'].cuda(rank)
            input_ids = batch['input_ids'].cuda(rank)
            attention_mask = batch['attention_mask'].cuda(rank)
            # Get output and calculate loss
            output = model(input_ids, attention_mask).logits
            loss = loss_func(output, labels)
            # Calculate loss and acc
            total_loss += loss.item()
            total_correct += calcuate_accuracy(output, labels)
            total_samples += labels.size(0)
            num_total += labels.size(0)
            average_loss = total_loss / num_total
            accuracy = total_correct / num_total * 100.0
        return average_loss, accuracy


def main(opts):
    print(f"Running basic DDP example on rank {opts.local_rank}.")
    # Set devices
    local_rank = opts.local_rank
    torch.cuda.set_device(local_rank)
    # Initialize process
    dist.init_process_group(backend='nccl')
    # Get dataloader and sampler
    train_loader, test_loader, train_sampler, test_sampler = dataloader(name="ag_news",
                                                                        token_name=opts.pretrained_model_name,
                                                                        train_length=10000,
                                                                        test_length=5000,
                                                                        batch_size=opts.batch_size,
                                                                        ddp=opts.ddp)
    # Get the model
    model = AutoModelForSequenceClassification.from_pretrained(opts.pretrained_model_name, num_labels=opts.num_classes)
    # Distribute model to GPUs
    model.to(local_rank)
    # Enable DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # Set the optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=opts.lr)
    num_training_steps = opts.epoch * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    # Initialize loss function of GPUs
    evaluation = torch.nn.CrossEntropyLoss().to(local_rank)
    # Training for epochs
    for epoch in range(opts.epoch):
        start_time = time.time()
        avg_train_loss, avg_train_acc = train(model, train_loader, optimizer, scheduler, evaluation, epoch, local_rank)
        end_time = time.time()
        if local_rank == 0:
            avg_test_loss, avg_test_acc = evaluate(model, test_loader, evaluation, epoch, local_rank)
        # calculate time
        epoch_time = end_time - start_time
        # Print results
        if epoch > 0:
            print("Epoch: ", epoch)
            print(f'\tTrain Loss: {avg_train_loss:.5f} | Train Acc: {avg_train_acc:.2f}%')
            if local_rank==0:
                print(f'\tTest Loss: {avg_test_loss:.5f} | Test Acc: {avg_test_acc:.2f}%')
            print(f"\tTime: {epoch_time:.2f} seconds")
        else:
            print("Warm Up train")


if __name__ == "__main__":
    # Get parser argument
    num_gpu = torch.cuda.device_count()
    print(f"You are using {num_gpu} GPUS!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased',
                        help='Name of the pre-trained BERT model')
    parser.add_argument('--epoch', type=int, default=3, help='Number of training epoches')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--lr', type=int, default=5e-5, help='Learning Rate')
    parser.add_argument('--pipe', type=bool, default=False, help='enable pipeline model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and testing')
    parser.add_argument('--ddp', action="store_true", help='enable distributed data parallel')
    parser.add_argument('--local-rank', default=-1, type=int, help='enable distributed data parallel')
    opts = parser.parse_args()
    # enable Distributed Data parallel
    main(opts=opts)
