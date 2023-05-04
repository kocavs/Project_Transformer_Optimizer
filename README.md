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

For distributed Data Parallel, run it by

CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 ddp.py --ddp --batch_size 32 --pretrained_model_name "roberta-base"


# Pretrained result with three models (With batch size 32 and one warm up epoch)
## bert-base-uncased
Epoch:  1<br>
        Train Loss: 0.00491 | Train Acc: 94.94%<br>
        Test. Loss: 0.00698 |  Test Acc: 92.90%<br>
        Time: 337.17 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00245 | Train Acc: 97.62%<br>
        Test. Loss: 0.00722 |  Test Acc: 93.50%<br>
        Time: 337.34 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
        Train Loss: 0.00499 | Train Acc: 94.89%<br>
        Test. Loss: 0.00660 |  Test Acc: 93.00%<br>
        Time: 173.95 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00280 | Train Acc: 97.42%<br>
        Test. Loss: 0.00722 |  Test Acc: 92.80%<br>
        Time: 173.88 seconds<br>

## roberta-base:
Epoch:  1<br>
        Train Loss: 0.00578 | Train Acc: 93.80%<br>
        Test. Loss: 0.00686 |  Test Acc: 93.40%<br>
        Time: 340.80 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00353 | Train Acc: 96.20%<br>
        Test. Loss: 0.00653 |  Test Acc: 94.40%<br>
        Time: 340.17 seconds<br>


# Using DP (Data Parallel) with three models (With batch size 32 and one warm up epoch)
## bert-base-uncased
Epoch:  1<br>
        Train Loss: 0.00489 | Train Acc: 95.05%<br>
        Test. Loss: 0.00645 |  Test Acc: 93.10%<br>
        Time: 199.22 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00257 | Train Acc: 97.53%<br>
        Test. Loss: 0.00658 |  Test Acc: 93.30%<br>
        Time: 199.30 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
        Train Loss: 0.00508 | Train Acc: 94.87%<br>
        Test. Loss: 0.00673 |  Test Acc: 92.80%<br>
        Time: 104.60 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00283 | Train Acc: 97.40%<br>
        Test. Loss: 0.00696 |  Test Acc: 93.50%<br>
        Time: 104.73 seconds<br>


## roberta-base:
Epoch:  1<br>
        Train Loss: 0.00571 | Train Acc: 93.76%<br>
        Test. Loss: 0.00642 |  Test Acc: 94.00%<br>
        Time: 113.75 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00341 | Train Acc: 96.58%<br>
        Test. Loss: 0.00677 |  Test Acc: 94.10%<br>
        Time: 113.67 seconds<br>
        
# Using DDP (Distributed Data Parallel) with three models (With batch size 32 and one warm up epoch)
## bert-base-uncased
Epoch:  1<br>
        Train Loss: 0.00491 | Train Acc: 95.18%<br>
        Test Loss: 0.00667 | Test Acc: 92.80%<br>
        Time: 149.79 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00309 | Train Acc: 96.92%<br>
        Test Loss: 0.00735 | Test Acc: 92.40%<br>
        Time: 149.67 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
        Train Loss: 0.00563 | Train Acc: 94.12%<br>
        Test Loss: 0.00776 | Test Acc: 92.00%<br>
        Time: 77.66 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00352 | Train Acc: 96.68%<br>
        Test Loss: 0.00806 | Test Acc: 91.60%<br>
        Time: 77.89 seconds<br>

## roberta-base:
Epoch:  1<br>
        Train Loss: 0.00544 | Train Acc: 94.44%<br>
        Test Loss: 0.00618 | Test Acc: 94.20%<br>
        Time: 151.49 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00351 | Train Acc: 96.62%<br>
        Test Loss: 0.00600 | Test Acc: 93.80%<br>
        Time: 151.53 seconds<br>

# Pretrained result with three models (With batch size 128 and one warm up epoch)
## bert-base-uncased
Epoch:  1<br>
        Train Loss: 0.00144 | Train Acc: 94.14%<br>
        Test. Loss: 0.00195 |  Test Acc: 91.48%<br>
        Time: 172.10 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00087 | Train Acc: 96.77%<br>
        Test. Loss: 0.00194 |  Test Acc: 92.12%<br>
        Time: 172.30 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
        Train Loss: 0.00192 | Train Acc: 92.14%<br>
        Test. Loss: 0.00218 |  Test Acc: 90.92%<br>
        Time: 88.13 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00128 | Train Acc: 95.25%<br>
        Test. Loss: 0.00199 |  Test Acc: 91.78%<br>
        Time: 88.29 seconds<br>

## roberta-base:
Epoch:  1<br>
        Train Loss: 0.00159 | Train Acc: 93.39%<br>
        Test. Loss: 0.00188 |  Test Acc: 92.86%<br>
        Time: 172.63 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00100 | Train Acc: 96.07%<br>
        Test. Loss: 0.00187 |  Test Acc: 93.16%<br>
        Time: 172.64 seconds<br>


# Using DP (Data Parallel) with three models (With batch size 128 and one warm up epoch)
## bert-base-uncased
Epoch:  1<br>
        Train Loss: 0.00153 | Train Acc: 93.88%<br>
        Test. Loss: 0.00197 |  Test Acc: 91.86%<br>
        Time: 171.70 seconds<br>
Epoch:  2
        Train Loss: 0.00097 | Train Acc: 96.40%<br>
        Test. Loss: 0.00195 |  Test Acc: 91.96%<br>
        Time: 171.77 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
        Train Loss: 0.00167 | Train Acc: 93.17%<br>
        Test. Loss: 0.00212 |  Test Acc: 91.74%<br>
        Time: 87.86 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00114 | Train Acc: 95.85%<br>
        Test. Loss: 0.00201 |  Test Acc: 91.76%<br>
        Time: 88.27 seconds<br>


## roberta-base:
Epoch:  1<br>
        Train Loss: 0.00149 | Train Acc: 93.93%<br>
        Test. Loss: 0.00178 |  Test Acc: 92.68%<br>
        Time: 172.27 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00096 | Train Acc: 96.17%<br>
        Test. Loss: 0.00195 |  Test Acc: 92.56%<br>
        Time: 172.65 seconds<br>

