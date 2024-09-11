'''
Resources used:
https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
https://stackoverflow.com/questions/73266661/multiple-input-model-with-pytorch-input-and-output-features
'''
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
# import concurrent.futures
from tqdm import tqdm
from typing import Final
import itertools
# from focal_loss import focal_loss

assert torch.cuda.is_available(), "Error: CUDA not available"
GPU = torch.device("cuda:0")

# Constant dict taken from colab notebook
ID_TO_NAME: Final[dict] = {
    0: 'actr_lag2_decay',
    1: 'adddriftbot2',
    2: 'addshiftbot3',
    3: 'antiflatbot',
    4: 'antirotnbot',
    5: 'biopic',
    6: 'boom',
    7: 'copybot',
    8: 'debruijn81',
    9: 'driftbot',
    10: 'flatbot3',
    11: 'foxtrotbot',
    12: 'freqbot2',
    13: 'granite',
    14: 'greenberg',
    15: 'halbot',
    16: 'inocencio',
    17: 'iocainebot',
    18: 'marble',
    19: 'markov5',
    20: 'markovbails',
    21: 'mixed_strategy',
    22: 'mod1bot',
    23: 'multibot',
    24: 'peterbot',
    25: 'phasenbott',
    26: 'pibot',
    27: 'piedra',
    28: 'predbot',
    29: 'r226bot',
    30: 'randbot',
    31: 'robertot',
    32: 'rockbot',
    33: 'rotatebot',
    34: 'russrocker4',
    35: 'shofar',
    36: 'sunCrazybot',
    37: 'sunNervebot',
    38: 'sweetrock',
    39: 'switchalot',
    40: 'switchbot',
    41: 'textbot',
    42: 'zq_move',
}
NAME_TO_ID: Final[dict] = {v: k for k, v in ID_TO_NAME.items()}

# Note: if any of these is altered, make sure to alter all of these parameters appropriately

# # For window size 200
# WINDOW_SIZE = 200   # Size of the window to use for the LSTM
# NUM_WINDOWS_PER_SERIES = 1000 - WINDOW_SIZE  # Number of windows per time series
# NUM_PROCS = 8        # Number of processes to use for multiprocessing
# SUBBATCH_SIZE = 100  # Size of subbatches to use for multiprocessing

# # For window size 200
# WINDOW_SIZE = 200   # Size of the window to use for the LSTM
# NUM_WINDOWS_PER_SERIES = 1000 - WINDOW_SIZE  # Number of windows per time series
# NUM_PROCS = 4        # Number of processes to use for multiprocessing
# SUBBATCH_SIZE = 200  # Size of subbatches to use for multiprocessing

# For window size 200
WINDOW_SIZE = 200   # Size of the window to use for the LSTM
NUM_WINDOWS_PER_SERIES = 1000 - WINDOW_SIZE  # Number of windows per time series
# NUM_PROCS = 1        # Number of processes to use for multiprocessing
# SUBBATCH_SIZE = 800  # Size of subbatches to use for multiprocessing

# assert NUM_PROCS * SUBBATCH_SIZE == NUM_WINDOWS_PER_SERIES, "Incorrect configuration of NUM_PROCS and SUBBATCH_SIZE"

class LSTM(nn.Module):
    """
    This model predicts the opponent's next move given 2 inputs:
    1. The first WINDOW_SIZE moves of the opponent and agent (agent acting purely randomly)
    2. The previous WINDOW_SIZE moves of the opponent and agent
    """
    def __init__(self):
        super().__init__()

        # lstmHiddenSize = 100
        # self.lstm = nn.LSTM(input_size=6, hidden_size=lstmHiddenSize, batch_first=True)
        # self.lstm.cuda()

        startingFirstActionSize = WINDOW_SIZE * 6 * 2
        self.firstResponsesNetwork = nn.Sequential(
            nn.Linear(startingFirstActionSize, startingFirstActionSize // 2),
            nn.ReLU(inplace=True),
            nn.Linear(startingFirstActionSize // 2, startingFirstActionSize // 4),
            nn.ReLU(inplace=True),
            nn.Linear(startingFirstActionSize // 4, startingFirstActionSize // 8),
            nn.ReLU(inplace=True),
            nn.Linear(startingFirstActionSize // 8, startingFirstActionSize // 16),
            nn.ReLU(inplace=True),
            nn.Linear(startingFirstActionSize // 16, startingFirstActionSize // 32),
            nn.ReLU(inplace=True),
            nn.Linear(startingFirstActionSize // 32, startingFirstActionSize // 64),
            nn.ReLU(inplace=True),
            nn.Linear(startingFirstActionSize // 64, 3),
            nn.Softmax(dim=1),
        )
        self.firstResponsesNetwork.cuda()

        # startingCombinedSize = lstmHiddenSize + startingFirstActionSize // 8
        # self.combinedNetwork = nn.Sequential(
        #     nn.Linear(startingCombinedSize, startingCombinedSize // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(startingCombinedSize // 2, startingCombinedSize // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(startingCombinedSize // 4, 3),
        #     nn.Softmax(dim=1),
        # )
        # self.combinedNetwork.cuda()

    def forward(self, recentActions, firstActions):
        x1, _ = self.lstm(recentActions)
        x1 = x1[:, -1, :]  # Get only the last output
        x2 = self.firstResponsesNetwork(firstActions)
        x = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim=1)
        x = self.combinedNetwork(x)
        return x

def load_data_from_folder(dir: str, train_split: float = 0.8):
    """Load data from all npy files in a folder and convert it to 2 tensors: 1 training, 1 testing."""
    paths = [os.path.join(dir, p) for p in os.listdir(dir) if p.endswith(".npy")]
    
    # Load data into single np array
    # data = [np.load(filepath)[:40] for filepath in paths]
    data = [np.load(filepath) for filepath in paths]
    assert all(d.shape == data[0].shape for d in data), "Expected all data to have same shape"
    
    ids = [[NAME_TO_ID[os.path.basename(p).split('-')[0]]] * len(data[0]) for p in paths]
    ids = np.array(list(itertools.chain.from_iterable(ids)))  # Flatten list of lists

    # Convert to one-hot encoding and return
    data = np.vstack(data)
    data = F.one_hot(torch.tensor(data).to(torch.int64), num_classes=3)
    assert len(data.shape) == 4, f"Expected 4 dimensional tensor, got {len(data.shape)} dimensional tensor"
    data = torch.reshape(data, (data.shape[0], data.shape[1], 6)).numpy().astype(np.uint8)
    assert len(data) == len(ids), f"Expected {len(ids)} time series, got {len(data)} data points"

    # Shuffle data and ids in the same order
    p = np.random.permutation(len(data))
    data = data[p]
    ids = ids[p]

    # Return 2 tuples of np arrays (each contains data and agent id): 1 for training, 1 for testing
    splitIndex = int(len(data) * train_split)
    return (data[:splitIndex], ids[:splitIndex]), (data[splitIndex:], ids[splitIndex:])

# This dataset creates the windows needed for the LSTM
class WindowDataset(data.Dataset):
    def __init__(self, data, labels, window_size=WINDOW_SIZE):
        assert len(data) == len(labels), f"Data and labels should have same length"
        self.data = data
        self.labels = labels
        self.window_size = window_size
    
    def shuffleData(self):
        """
        Call this before each epoch to shuffle data.
        Uses permutation to shuffle data and labels in the same order.
        """
        p = np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.labels = self.labels[p]

    def __len__(self):
        return len(self.data)
    
    # def getItem(self, index, subbatch_number: int, subbatch_size: int = SUBBATCH_SIZE):
    #     timeSeries = self.data[index]
    #     label = self.labels[index]
    #     startIndex = subbatch_size*subbatch_number
    #     X1 = np.array([timeSeries[i:i+self.window_size] for i in range(startIndex, startIndex+subbatch_size)])
    #     X2 = np.array([timeSeries[:self.window_size] for _ in range(subbatch_size)])
    #     X2 = np.reshape(X2, (X2.shape[0], np.prod(X2.shape[1:])))
    #     y = np.array([timeSeries[i+self.window_size, :3] for i in range(startIndex, startIndex+subbatch_size)])
    #     return torch.tensor(X1, dtype=torch.float32, device=GPU), \
    #         torch.tensor(X2, dtype=torch.float32, device=GPU), \
    #         torch.tensor(y, dtype=torch.float32, device=GPU), label

    # def getItemNoSubbatch(self, index):
    #     timeSeries = self.data[index]
    #     label = self.labels[index]
    #     X1 = np.array([timeSeries[i:i+self.window_size] for i in range(NUM_WINDOWS_PER_SERIES)])
    #     X2 = np.array([timeSeries[:self.window_size] for _ in range(NUM_WINDOWS_PER_SERIES)])
    #     X2 = np.reshape(X2, (X2.shape[0], np.prod(X2.shape[1:])))
    #     y = np.array([timeSeries[i+self.window_size, :3] for i in range(NUM_WINDOWS_PER_SERIES)])
    #     return torch.tensor(X1, dtype=torch.float32, device=GPU), \
    #         torch.tensor(X2, dtype=torch.float32, device=GPU), \
    #         torch.tensor(y, dtype=torch.float32, device=GPU), label

    def __getitem__(self, index):
        timeSeries = self.data[index]
        label = self.labels[index]
        X1 = np.array([timeSeries[i:i+self.window_size] for i in range(NUM_WINDOWS_PER_SERIES)])
        X2 = np.array([timeSeries[:self.window_size] for _ in range(NUM_WINDOWS_PER_SERIES)])
        X2 = np.reshape(X2, (X2.shape[0], np.prod(X2.shape[1:])))
        y = np.array([timeSeries[i+self.window_size, :3] for i in range(NUM_WINDOWS_PER_SERIES)])
        return torch.tensor(X1, dtype=torch.float32, device=GPU), \
            torch.tensor(X2, dtype=torch.float32, device=GPU), \
            torch.tensor(y, dtype=torch.float32, device=GPU), label

def iterateOverDataset(dataset: WindowDataset):
    """
    Generator used to iterate over a dataset.

    Args:
        dataset: The dataset to iterate over (either train or validation set)
    """
    # procNums = list(range(NUM_PROCS))
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for batchNum in tqdm(range(len(dataset))):
    #         for X1_batch, X2_batch, y_batch, label in executor.map(
    #                 dataset.getItem, [batchNum]*NUM_PROCS, procNums):
    #             yield X1_batch, X2_batch, y_batch, label
    for batchNum in tqdm(range(len(dataset))):
        yield dataset.getItemNoSubbatch(batchNum)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2) -> None:
        super().__init__()
        self.gamma = gamma
    
    def forward(self, outputs, targets):
        # Below code copied from here: https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss

if __name__ == '__main__':
    VALIDATE_EVERY = 2  # Validate every n epochs
    n_epochs = 31       # Number of epochs to train for
    # saveDir = f"D:/cs486/allModels_wsize{WINDOW_SIZE}/"  # Where to save results
    saveDir = f"D:/cs486/allModels_focal_wsize{WINDOW_SIZE}/"  # Where to save results
    
    os.makedirs(saveDir, exist_ok=False)  # exist_ok=False to avoid overwriting existing results

    trainLoaded, testLoaded = load_data_from_folder("AllBots_vs_random/")
    trainData, trainLabels = trainLoaded
    testData, testLabels = testLoaded

    np.save(os.path.join(saveDir, "trainData.npy"), trainData)
    np.save(os.path.join(saveDir, "trainLabels.npy"), trainLabels)
    np.save(os.path.join(saveDir, "testData.npy"), testData)
    np.save(os.path.join(saveDir, "testLabels.npy"), testLabels)

    # Track number of each label in train and validation data
    trainCounts = np.bincount(trainLabels)
    testCounts = np.bincount(testLabels)
    print(f"Train counts: {[f'{ID_TO_NAME[i]}: {trainCounts[i]}' for i in range(len(ID_TO_NAME))]}")
    print(f"Test counts: {[f'{ID_TO_NAME[i]}: {testCounts[i]}' for i in range(len(ID_TO_NAME))]}")

    model = LSTM()
    optimizer = optim.Adam(model.parameters())
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = focal_loss(gamma=2, device="cuda:0")
    loss_fn = FocalLoss()
    dataset = WindowDataset(trainData, trainLabels, window_size=WINDOW_SIZE)
    valSet = WindowDataset(testData, testLabels, window_size=WINDOW_SIZE)

    trainLoader = data.DataLoader(dataset, batch_size=None, batch_sampler=None, 
                                  shuffle=True, num_workers=1, prefetch_factor=32)
    valLoader = data.DataLoader(valSet, batch_size=None, batch_sampler=None, 
                                shuffle=False, num_workers=1, prefetch_factor=32)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        numCorrect = [0] * len(ID_TO_NAME)
        # dataset.shuffleData()
        model.train()

        # for X1_batch, X2_batch, y_batch, label in iterateOverDataset(dataset):
        # for X1_batch, X2_batch, y_batch, label in trainLoader:
        for _, (X1_batch, X2_batch, y_batch, label) in tqdm(enumerate(trainLoader, 0), unit="batch", total=len(trainLoader)):
            # Batch size here is SUBBATCH_SIZE, reduce this if running out of memory
            y_pred = model(X1_batch, X2_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                numCorrect[label] += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).int().sum()

        numWindowsInTrain = len(dataset) * NUM_WINDOWS_PER_SERIES

        # Validation done every VALIDATE_EVERY epochs
        if epoch % VALIDATE_EVERY != 0:
            print(f"Epoch {epoch}: train accuracy {sum(numCorrect) / numWindowsInTrain :.2%}")
            continue
        model.eval()
        numCorrectTest = [0] * len(ID_TO_NAME)
        with torch.no_grad():
            # for X1_batch, X2_batch, y_batch, label in iterateOverDataset(valSet):
            for _, (X1_batch, X2_batch, y_batch, label) in tqdm(enumerate(valLoader, 0), unit="batch", total=len(valLoader)):
                y_pred = model(X1_batch, X2_batch)
                numCorrectTest[label] += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).int().sum()
        numWindowsInVal = len(valSet) * NUM_WINDOWS_PER_SERIES
        print(f"Epoch {epoch}: train accuracy {sum(numCorrect) / numWindowsInTrain :.2%}, test accuracy {sum(numCorrectTest) / numWindowsInVal :.2%}")

        # Save model
        directoryPath = os.path.join(saveDir, str(epoch))
        os.makedirs(directoryPath, exist_ok=False)  # exist_ok=False to avoid overwriting existing results
        torch.save(model.state_dict(), os.path.join(directoryPath, "model_stateDict.pt"))
        torch.save(model, os.path.join(directoryPath, "model.pt"))

        # Save detailed accuracy metrics in a text file
        with open(os.path.join(directoryPath, "accuracy.txt"), "w") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"Train accuracy {sum(numCorrect) / numWindowsInTrain :.2%}, test accuracy {sum(numCorrectTest) / numWindowsInVal :.2%}\n")
            f.write(f"Train dataset num windows: {numWindowsInTrain}, test dataset num windows: {numWindowsInVal}\n")
            f.write(f"Train number windows correct: {sum(numCorrect)}, test number windows correct: {sum(numCorrectTest)}\n\n")

            f.write("Detailed accuracy metrics:\n")
            for i, name in ID_TO_NAME.items():
                f.write(f"{name}: {numCorrect[i] / (trainCounts[i] * NUM_WINDOWS_PER_SERIES) :.2%} train, ")
                f.write(f"{numCorrectTest[i] / (testCounts[i] * NUM_WINDOWS_PER_SERIES) :.2%} test\n")
            
            f.write("\nDetailed counts:\n")
            for i, name in ID_TO_NAME.items():
                f.write(f"{name}: {trainCounts[i]} train series, {testCounts[i]} test series\n")
