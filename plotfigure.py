import matplotlib.pyplot as plt
import numpy as np


def plot32batchsize():
    # Generate random data for lists A, B, and C
    K = np.array(["1_GPU", "2_GPUs(DP)", "2_GPUs(DDP)"])
    A = np.array([[(339.59+338.74)/2, (199.22+199.30)/2, (149.79+149.67)/2],
                  [(340.50+340.03)/2, (113.75+113.67)/2, (151.49+151.53)/2],
                  [(174.58+173.88)/2, (104.60+104.73)/2, (77.66+77.89)/2]])
    A = A.T
    D = [-0.1, 0, 0.1]
    # Create a figure with three subplots
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Plot the data for list A on the first subplot
    x = np.arange(len(K))
    width = 0.1
    ax1.bar(x - width, A[:, 0], width=width, label="bert-base-uncased")
    ax1.bar(x, A[:, 1], width=width, label="roberta-base")
    ax1.bar(x + width, A[:, 2], width=width, label="distilbert-base-uncased")
    for i in range(len(K)):
        for j in range(len(K)):
            plt.text(i+D[j], round(A[i][j]), str(round(A[i][j])), ha='center', va='bottom')
    ax1.set_xticks(x)
    ax1.set_xticklabels(K)
    ax1.set_xlabel("Training methods")
    ax1.set_ylabel("Time to execute (log scale)")
    ax1.set_title("Training time with 3 methods in batch size 32")
    ax1.legend()
    # Show the plot
    plt.show()


def plotaccloss(name, data, name2, data2):
    epoch = [1, 2]
    # Initialize figure with 2 ax
    fig, (ax, ax2) = plt.subplots(1, 2)
    # plot bar figure
    ax.plot(epoch, data[0])
    ax.plot(epoch, data[1])
    ax.plot(epoch, data[2])
    ax2.plot(epoch, data2[0])
    ax2.plot(epoch, data2[1])
    ax2.plot(epoch, data2[2])
    ax.set_xticks(epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(name)
    ax.set_title("Epoch vs "+name)
    ax2.set_xticks(epoch)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(name2)
    ax2.set_title("Epoch vs " + name2)
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def plot128batchsize():
    # Generate random data for lists A, B, and C
    K = np.array(["2_GPUs(DP)", "2_GPUs(DDP)+Mixed-precision"])
    A = np.array([[(173.57+172.01)/2, (83.43+81.23)/2],
                  [(174.80+172.94)/2, (84.50+82.26)/2],
                  [(90.32+88.44)/2, (42.36+39.95)/2]])
    A = A.T
    D = [-0.1, 0.0, 0.1]
    # Create a figure with three subplots
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Plot the data for list A on the first subplot
    x = np.arange(len(K))
    width = 0.1
    ax1.bar(x - width, A[:, 0], width=width, label="bert-base-uncased")
    ax1.bar(x, A[:, 1], width=width, label="roberta-base")
    ax1.bar(x + width, A[:, 2], width=width, label="distilbert-base-uncased")
    for i in range(len(K)):
        for j in range(len(K)+1):
            plt.text(i+D[j], round(A[i][j]), str(round(A[i][j])), ha='center', va='bottom')
    ax1.set_xticks(x)
    ax1.set_xticklabels(K)
    ax1.set_xlabel("Training methods")
    ax1.set_ylabel("Time to execute (log scale)")
    ax1.set_title("Training time with enhanced methods in batch size 128")
    ax1.legend()
    # Show the plot
    plt.show()


plot32batchsize()
plot128batchsize()
plotaccloss(name="Acc",
            data=[[93.10, 93.30], [92.80, 93.50], [94.00, 94.10]],
            name2="Loss",
            data2=[[0.00645, 0.00658], [0.00673, 0.00696], [0.00642, 0.00677]])
