import os
import argparse
import torch
import time

from main import train, evaluate
from DataLoad import dataloader

from transformers import get_scheduler
from transformers import RobertaConfig, RobertaForSequenceClassification
import deepspeed
from tqdm import tqdm

# Function to create the pipeline model
def create_pipeline_model(device, config_path="deepspeed_config.json"):

    config = RobertaConfig.from_pretrained('roberta-base', num_labels=4)
    model = RobertaForSequenceClassification(config)

    model = model.to(device)
    
    model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_path)

    return model, optimizer

def main():
    # Define the arguments
    parser = argparse.ArgumentParser(description='Train a RoBERTa model with pipeline parallelism on the AG News dataset.')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='Logging directory')
    parser.add_argument('--eval_steps', type=int, default=50, help='Evaluation steps')

    args = parser.parse_args()

    # Set the devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, _, _ = dataloader(name="ag_news", token_name='roberta-base', train_length=10000, test_length=5000, batch_size=args.train_batch_size)
    # Create the pipeline model
    model, optimizer= create_pipeline_model(device)

    num_training_steps = args.num_train_epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )
    
    for epoch in range(args.train_batch_size):
        start_time = time.time()
        
        avg_train_loss, avg_train_acc = train(model, train_loader, optimizer, scheduler)
        
        end_time = time.time()
        
        avg_test_loss, avg_test_acc = evaluate(model, test_loader)
        
        epoch_time = end_time - start_time
        
        if epoch > 0:
            print("Epoch: ", epoch)
            print(f'\tTrain Loss: {avg_train_loss:.5f} | Train Acc: {avg_train_acc:.2f}%')
            print(f'\tTest. Loss: {avg_test_loss:.5f} |  Test Acc: {avg_test_acc:.2f}%')
            print(f"\tTime: {epoch_time:.2f} seconds")
        else:
            print("Warm Up train")

if __name__ == '__main__':
    main()
