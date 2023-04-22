from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)


def dataloader(name, num_train, num_test, batch_size):
    # Download dataset
    datasets = load_dataset("glue", name, cache_dir="./dataset")
    # Tokenizing dataset
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # print("train_datasets: \n", tokenized_datasets)
    train_datasets = tokenized_datasets["train"].select(range(num_train))
    test_datasets = tokenized_datasets["test"].select(range(num_test))

    # print("train_datasets: \n", train_datasets)
    # print(train_datasets["input_ids"].shape)

    # Here we shuffle our train dataloader
    train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_datasets, batch_size=batch_size)

    return train_dataloader, test_dataloader


# dataloader(name="cola", num_train=1000, num_test=1000, batch_size=64)
