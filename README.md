# Project_Transformer_Optimizer

# Requirement

1. transformer
2. datasets

# Executing the project:
For base and Data Parallel, run it by
```
python main.py
```
        options:
        -h, --help            show this help message and exit
        --pretrained_model_name PRETRAINED_MODEL_NAME Name of the pre-trained model
        --epoch EPOCH         Number of training epoches
        --num_classes NUM_CLASSES Number of classes
        --lr LR               Learning Rate
        --DP DP               enable data parallel
        --pipe PIPE           enable pipeline model
        --batch_size BATCH_SIZE Batch size for training and testing

For Distributed Data Parallel, run it by
```
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 ddp.py --ddp --batch_size 32 --pretrained_model_name "roberta-base"
```

# Pretrained result with three models (With batch size 32 and warm up steps)
## bert-base-uncased
Epoch:  1
        Train Loss: 0.01392 | Train Acc: 84.04%<br>
        Test. Loss: 0.00820 |  Test Acc: 91.03%<br>
        Time: 339.59 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00489 | Train Acc: 94.90%<br>
        Test. Loss: 0.00802 |  Test Acc: 91.26%<br>
        Time: 338.74 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
        Train Loss: 0.01435 | Train Acc: 85.00%<br>
        Test. Loss: 0.00859 |  Test Acc: 90.57%<br>
        Time: 174.58 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00518 | Train Acc: 94.70%<br>
        Test. Loss: 0.00787 |  Test Acc: 91.67%<br>
        Time: 173.88 seconds<br>

## roberta-base:
Epoch:  1<br>
        Train Loss: 0.01373 | Train Acc: 83.60%<br>
        Test. Loss: 0.00804 |  Test Acc: 91.30%<br>
        Time: 340.50 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00506 | Train Acc: 94.55%<br>
        Test. Loss: 0.00732 |  Test Acc: 92.29%<br>
        Time: 340.03 seconds<br>


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


# Using DP (Data Parallel) with three models (With batch size 128 and warm up steps)
## bert-base-uncased
Epoch:  1<br>
        Train Loss: 0.00452 | Train Acc: 79.25%<br>
        Test. Loss: 0.00224 |  Test Acc: 90.55%<br>
        Time: 173.57 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00162 | Train Acc: 93.09%<br>
        Test. Loss: 0.00208 |  Test Acc: 90.95%<br>
        Time: 172.01 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
        Train Loss: 0.00470 | Train Acc: 80.26%<br>
        Test. Loss: 0.00239 |  Test Acc: 89.76%<br>
        Time: 90.32 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00198 | Train Acc: 92.15%<br>
        Test. Loss: 0.00215 |  Test Acc: 90.74%<br>
        Time: 88.44 seconds<br>


## roberta-base:
Epoch:  1<br>
        Train Loss: 0.00460 | Train Acc: 76.74%<br>
        Test. Loss: 0.00225 |  Test Acc: 90.62%<br>
        Time: 174.80 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00164 | Train Acc: 92.89%<br>
        Test. Loss: 0.00196 |  Test Acc: 91.82%<br>
        Time: 172.94 seconds<br>

# Using DP(Data Parallel) + mixed-precision training with three models (With batch size 128 and warm up steps)

## bert-base-uncased
Epoch:  1<br>
        Train Loss: 0.00465 | Train Acc: 79.36%<br>
        Test. Loss: 0.00231 |  Test Acc: 89.92%<br>
        Time: 83.43 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00172 | Train Acc: 93.17%<br>
        Test. Loss: 0.00211 |  Test Acc: 90.84%<br>
        Time: 81.23 seconds<br>

## distilbert-base-uncased:

Epoch:  1<br>
        Train Loss: 0.00473 | Train Acc: 78.55%<br>
        Test. Loss: 0.00226 |  Test Acc: 90.47%<br>
        Time: 42.36 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00191 | Train Acc: 92.12%<br>
        Test. Loss: 0.00215 |  Test Acc: 90.57%<br>
        Time: 39.95 seconds<br>

## roberta-base:

Epoch:  1<br>
        Train Loss: 0.00442 | Train Acc: 77.98%<br>
        Test. Loss: 0.00220 |  Test Acc: 90.71%<br>
        Time: 84.50 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00183 | Train Acc: 92.25%<br>
        Test. Loss: 0.00204 |  Test Acc: 91.43%<br>
        Time: 82.26 seconds<br>