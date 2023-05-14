# Project_Transformer_Optimizer
Currently, there are numerous large pre-trained models based on transformers. 
However, these models often require a significant amount of time to finetune. 
Our task is to decrease the training time for these large models by using various 
parallel methods while ensuring that their quality remains unaffected. To demonstrate 
the feasibility of the parallel approach, we will test several models and compare the 
time saved for each model after optimization. This will enable us to observe and measure 
the impact of parallel methods on different models.

# Repository and code structure:
* main.py            (Base method, DP method and mixed method)
* DataLoad.py        (Dataloader for main.py)
* README.md          (Readme file contains introduction and results)
* ddp.py             (DDP method)
* plotfigure.py      (Code to plot image results)

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
        --mixed Mixed-precision training          enable Mixed-precision training
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


# Using DP (Data Parallel) with three models (With batch size 32 and warm up steps)
## bert-base-uncased
Epoch:  1<br>
	Train Loss: 0.01359 | Train Acc: 84.14%<br>
	Test. Loss: 0.00865 |  Test Acc: 90.82%<br>
	Time: 201.73 seconds<br>
Epoch:  2<br>
	Train Loss: 0.00457 | Train Acc: 95.17%<br>
	Test. Loss: 0.00791 |  Test Acc: 91.43%<br>
	Time: 198.37 seconds<br>

## distilbert-base-uncased:
Epoch:  1<br>
	Train Loss: 0.01427 | Train Acc: 84.38%<br>
	Test. Loss: 0.00819 |  Test Acc: 91.04%<br>
	Time: 107.80 seconds<br>
Epoch:  2<br>
	Train Loss: 0.00520 | Train Acc: 94.83%<br>
	Test. Loss: 0.00802 |  Test Acc: 91.47%<br>
	Time: 103.65 seconds<br>


## roberta-base:
Epoch:  1<br>
	Train Loss: 0.01468 | Train Acc: 82.24%<br>
	Test. Loss: 0.00921 |  Test Acc: 90.49%<br>
	Time: 202.26 seconds<br>
Epoch:  2<br>
	Train Loss: 0.00602 | Train Acc: 93.38%<br>
	Test. Loss: 0.00775 |  Test Acc: 91.76%<br>
	Time: 201.82 seconds<br>
        
# Using DDP (Distributed Data Parallel) with three models (With batch size 32 and warm up steps)
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
        Train Loss: 0.00528 | Train Acc: 94.36%<br>
        Test Loss: 0.00646 | Test Acc: 93.60%<br>
        Time: 174.24 seconds<br>
Epoch:  2<br>
        Train Loss: 0.00333 | Train Acc: 96.64%<br>
        Test Loss: 0.00628 | Test Acc: 94.20%<br>
        Time: 174.46 seconds<br>


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
