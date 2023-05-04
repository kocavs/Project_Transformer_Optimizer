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


# Using DP (Data Parallel) with three models (With batch size 32 and one warm up epoch)
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



# Pretrained result with three models (With batch size 128 and one warm up epoch)
## bert-base-uncased
Epoch:  1
        Train Loss: 0.00144 | Train Acc: 94.14%
        Test. Loss: 0.00195 |  Test Acc: 91.48%
        Time: 172.10 seconds
Epoch:  2
        Train Loss: 0.00087 | Train Acc: 96.77%
        Test. Loss: 0.00194 |  Test Acc: 92.12%
        Time: 172.30 seconds

## distilbert-base-uncased:
Epoch:  1
        Train Loss: 0.00192 | Train Acc: 92.14%
        Test. Loss: 0.00218 |  Test Acc: 90.92%
        Time: 88.13 seconds
Epoch:  2
        Train Loss: 0.00128 | Train Acc: 95.25%
        Test. Loss: 0.00199 |  Test Acc: 91.78%
        Time: 88.29 seconds

## roberta-base:
Epoch:  1
        Train Loss: 0.00159 | Train Acc: 93.39%
        Test. Loss: 0.00188 |  Test Acc: 92.86%
        Time: 172.63 seconds
Epoch:  2
        Train Loss: 0.00100 | Train Acc: 96.07%
        Test. Loss: 0.00187 |  Test Acc: 93.16%
        Time: 172.64 seconds


# Using DP (Data Parallel) with three models (With batch size 128 and one warm up epoch)
## bert-base-uncased
Epoch:  1
        Train Loss: 0.00153 | Train Acc: 93.88%
        Test. Loss: 0.00197 |  Test Acc: 91.86%
        Time: 171.70 seconds
Epoch:  2
        Train Loss: 0.00097 | Train Acc: 96.40%
        Test. Loss: 0.00195 |  Test Acc: 91.96%
        Time: 171.77 seconds

## distilbert-base-uncased:
Epoch:  1
        Train Loss: 0.00167 | Train Acc: 93.17%
        Test. Loss: 0.00212 |  Test Acc: 91.74%
        Time: 87.86 seconds
Epoch:  2
        Train Loss: 0.00114 | Train Acc: 95.85%
        Test. Loss: 0.00201 |  Test Acc: 91.76%
        Time: 88.27 seconds


## roberta-base:
Epoch:  1
        Train Loss: 0.00149 | Train Acc: 93.93%
        Test. Loss: 0.00178 |  Test Acc: 92.68%
        Time: 172.27 seconds
Epoch:  2
        Train Loss: 0.00096 | Train Acc: 96.17%
        Test. Loss: 0.00195 |  Test Acc: 92.56%
        Time: 172.65 seconds



## pipline running requirement
        1. export LD_LIBRARY_PATH=/lib:/lib64:/usr/lib:/share/apps/openmpi/4.0.5/gcc/lib:${LD_LIBRARY_PATH}
        2. pip install mpi4py