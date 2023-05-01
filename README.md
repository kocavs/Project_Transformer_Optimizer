# Project_Transformer_Optimizer

# Requirement

1. transformer
2. datasets

# Executing the project:

python main.py

        options:
        -h, --help            show this help message and exit
        --pretrained_model_name PRETRAINED_MODEL_NAME Name of the pre-trained model
        --epoch EPOCH         Number of training epoches
        --num_classes NUM_CLASSES Number of classes
        --lr LR               Learning Rate
        --DP DP               enable data parallel
        --pipe PIPE           enable pipeline model
        --batch_size BATCH_SIZE Batch size for training and testing

# Pretrained result with three models (With batch size 32 and one warm up epoch)
## bert-base-uncased
Epoch:  1
        Train Loss: 0.00491 | Train Acc: 94.94%
        Test. Loss: 0.00698 |  Test Acc: 92.90%
        Time: 337.17 seconds
Epoch:  2
        Train Loss: 0.00245 | Train Acc: 97.62%
        Test. Loss: 0.00722 |  Test Acc: 93.50%
        Time: 337.34 seconds

## distilbert-base-uncased:
Epoch:  1
        Train Loss: 0.00499 | Train Acc: 94.89%
        Test. Loss: 0.00660 |  Test Acc: 93.00%
        Time: 173.95 seconds
Epoch:  2
        Train Loss: 0.00280 | Train Acc: 97.42%
        Test. Loss: 0.00722 |  Test Acc: 92.80%
        Time: 173.88 seconds

## roberta-base:
Epoch:  1
        Train Loss: 0.00578 | Train Acc: 93.80%
        Test. Loss: 0.00686 |  Test Acc: 93.40%
        Time: 340.80 seconds
Epoch:  2
        Train Loss: 0.00353 | Train Acc: 96.20%
        Test. Loss: 0.00653 |  Test Acc: 94.40%
        Time: 340.17 seconds

# Using DDP (Data Parallel) with three models (With batch size 32 and one warm up epoch)
## bert-base-uncased
Epoch:  1
        Train Loss: 0.00489 | Train Acc: 95.05%
        Test. Loss: 0.00645 |  Test Acc: 93.10%
        Time: 199.22 seconds
Epoch:  2
        Train Loss: 0.00257 | Train Acc: 97.53%
        Test. Loss: 0.00658 |  Test Acc: 93.30%
        Time: 199.30 seconds

## distilbert-base-uncased:
Epoch:  1
        Train Loss: 0.00508 | Train Acc: 94.87%
        Test. Loss: 0.00673 |  Test Acc: 92.80%
        Time: 104.60 seconds
Epoch:  2
        Train Loss: 0.00283 | Train Acc: 97.40%
        Test. Loss: 0.00696 |  Test Acc: 93.50%
        Time: 104.73 seconds


## roberta-base:
Epoch:  1
        Train Loss: 0.00571 | Train Acc: 93.76%
        Test. Loss: 0.00642 |  Test Acc: 94.00%
        Time: 113.75 seconds
Epoch:  2
        Train Loss: 0.00341 | Train Acc: 96.58%
        Test. Loss: 0.00677 |  Test Acc: 94.10%
        Time: 113.67 seconds
