# Tutorial used: https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import concurrent.futures
from tqdm import tqdm

# Note: if any of these is altered, make sure to alter all of these parameters appropriately

# For window size 200
WINDOW_SIZE = 200   # Size of the window to use for the LSTM
NUM_WINDOWS_PER_SERIES = 1000 - WINDOW_SIZE  # Number of windows per time series
NUM_PROCS = 8        # Number of processes to use for multiprocessing
SUBBATCH_SIZE = 100  # Size of subbatches to use for multiprocessing

# For window size 100
# WINDOW_SIZE = 100   # Size of the window to use for the LSTM
# NUM_WINDOWS_PER_SERIES = 1000 - WINDOW_SIZE  # Number of windows per time series
# NUM_PROCS = 9        # Number of processes to use for multiprocessing
# SUBBATCH_SIZE = 100  # Size of subbatches to use for multiprocessing

# For window size 50
# WINDOW_SIZE = 50   # Size of the window to use for the LSTM
# NUM_WINDOWS_PER_SERIES = 1000 - WINDOW_SIZE  # Number of windows per time series
# NUM_PROCS = 10        # Number of processes to use for multiprocessing
# SUBBATCH_SIZE = 95  # Size of subbatches to use for multiprocessing

assert NUM_PROCS * SUBBATCH_SIZE == NUM_WINDOWS_PER_SERIES, "Incorrect configuration of NUM_PROCS and SUBBATCH_SIZE"

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, batch_first=True)
        self.linear = nn.Linear(50, 3)
        self.softmax = nn.Softmax(dim=1)
        # self.lstm = nn.LSTM(input_size=6, hidden_size=100, batch_first=True)
        # self.linear = nn.Linear(100, 25)
        # self.linear2 = nn.Linear(25, 3)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x)
        x, _ = self.lstm(x.to(torch.float32))
        x = self.linear(x[:, -1, :])  # Get only the last output
        # x = self.linear2(x)
        x = self.softmax(x)
        return x

def load_data_from_folder(dir: str, train_split: float = 0.8):
    """Load data from all npy files in a folder and convert it to 2 tensors: 1 training, 1 testing."""
    paths = [os.path.join(dir, p) for p in os.listdir(dir) if p.endswith(".npy")]
    
    # Load data into single np array
    data = [np.load(filepath) for filepath in paths]
    data = np.vstack(data)

    # Convert to one-hot encoding and return
    data = F.one_hot(torch.tensor(data).to(torch.int64), num_classes=3)
    assert len(data.shape) == 4, f"Expected 4 dimensional tensor, got {len(data.shape)} dimensional tensor"
    data = torch.reshape(data, (data.shape[0], data.shape[1], 6)).numpy().astype(np.uint8)

    # Return 2 tensors: 1 for training, 1 for testing
    splitIndex = int(len(data) * train_split)
    return data[:splitIndex], data[splitIndex:]

# This dataset creates the windows needed for the LSTM
class WindowDataset(data.Dataset):
    def __init__(self, data, window_size=200):
        self.data = data
        self.window_size = window_size
    
    def shuffleData(self):
        """Call this before each epoch to shuffle data."""
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def getItem(self, index, subbatch_number: int, subbatch_size: int = SUBBATCH_SIZE):
        timeSeries = self.data[index]
        startIndex = subbatch_size*subbatch_number
        X = [timeSeries[i:i+self.window_size] for i in range(startIndex, startIndex+subbatch_size)]
        y = [timeSeries[i+self.window_size, :3] for i in range(startIndex, startIndex+subbatch_size)]
        return torch.tensor(np.array(X)), torch.tensor(np.array(y))
    
    # def __getitem__(self, index):
    #     timeSeries = self.data[index]  # Get a single time series
    #     X = [timeSeries[i:i+self.window_size] for i in range(0, 1000-self.window_size)]
    #     y = [timeSeries[i+self.window_size, :3] for i in range(0, 1000-self.window_size)]
    #     return torch.tensor(np.array(X)), torch.tensor(np.array(y))

# def iterateOverDataset(dataset: WindowDataset, subbatches: int = 8, subbatch_size: int = 100):
#     """Iterate over a dataset."""
#     for batchNum in range(len(dataset)):
#         X, y = dataset[batchNum]
#         print(batchNum, '/', len(dataset))
#         yield X, y
#         for subBatchNum in range(subbatches):
#             X_batch = X[subbatch_size*subBatchNum:subbatch_size*(subBatchNum+1)]
#             y_batch = y[subbatch_size*subBatchNum:subbatch_size*(subBatchNum+1)]
#             yield X_batch, y_batch

def iterateOverDataset(dataset: WindowDataset):
    """
    Generator used to iterate over a dataset.
    
    The raw dataset consists of a list of time series. Each time series is a 2D array of shape (1000, 6).
    This generator converts each time series into a list of windows of shape (window_size, 6). It will yield
    NUM_WINDOWS_PER_SERIES windows for each time series. It uses multiprocessing to speed up the process 
    (since creating these time series is the bottleneck in training, especially with a GPU). This cannot be
    done before training since all the individual time series are too large to fit in memory (which is also why
    a generator is used instead of a list).

    Args:
        dataset: The dataset to iterate over (either train or validation set)
    """
    procNums = list(range(NUM_PROCS))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for batchNum in tqdm(range(len(dataset))):
            for X_batch, y_batch in executor.map(dataset.getItem, [batchNum]*NUM_PROCS, procNums):
                # print(X_batch.shape, y_batch.shape)
                yield X_batch, y_batch
    # for batchNum in range(len(dataset)):
    #     X, y = dataset[batchNum]
    #     print(batchNum, '/', len(dataset))
    #     yield X, y
    #     for subBatchNum in range(subbatches):
    #         X_batch = X[subbatch_size*subBatchNum:subbatch_size*(subBatchNum+1)]
    #         y_batch = y[subbatch_size*subBatchNum:subbatch_size*(subBatchNum+1)]
    #         yield X_batch, y_batch


if __name__ == '__main__':
    VALIDATE_EVERY = 2  # Validate every n epochs
    n_epochs = 21       # Number of epochs to train for
    saveDir = f"D:/cs486/lstm_models_fullData_wsize{WINDOW_SIZE}_softmax/"  # Where to save results
    
    os.makedirs(saveDir, exist_ok=False)  # exist_ok=False to avoid overwriting existing results

    trainData, testData = load_data_from_folder("Greenberg_data_full/")

    model = LSTM()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    dataset = WindowDataset(trainData, window_size=WINDOW_SIZE)
    valSet = WindowDataset(testData, window_size=WINDOW_SIZE)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        numCorrect = 0
        dataset.shuffleData()
        model.train()

        for X_batch, y_batch in iterateOverDataset(dataset):
            # Batch size here is SUBBATCH_SIZE, reduce this if running out of memory
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                numCorrect += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().sum()
        
        # Validation done every VALIDATE_EVERY epochs
        if epoch % VALIDATE_EVERY != 0:
            print(f"Epoch {epoch}: train accuracy {100 * numCorrect / (len(dataset) * 800)}%")
            continue
        model.eval()
        numCorrectTest = 0
        with torch.no_grad():
            for X_batch, y_batch in iterateOverDataset(valSet):
                y_pred = model(X_batch)
                numCorrectTest += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().sum()
        numWindowsInVal = len(valSet) * NUM_WINDOWS_PER_SERIES
        numWindowsInTrain = len(dataset) * NUM_WINDOWS_PER_SERIES
        print(f"Epoch {epoch}: train accuracy {100 * numCorrect / numWindowsInTrain}%, test accuracy {100 * numCorrectTest / numWindowsInVal}%")

        # Save model
        directoryPath = os.path.join(saveDir, str(epoch))
        os.makedirs(directoryPath, exist_ok=False)  # exist_ok=False to avoid overwriting existing results
        torch.save(model.state_dict(), os.path.join(directoryPath, "model_stateDict.pt"))
        torch.save(model, os.path.join(directoryPath, "model.pt"))

        # Save accuracy in a text file
        with open(os.path.join(directoryPath, "accuracy.txt"), "w") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"Train accuracy {100 * numCorrect / numWindowsInTrain}%, test accuracy {100 * numCorrectTest / numWindowsInVal}%\n")
            f.write(f"Train dataset size: {numWindowsInTrain}, test dataset size: {numWindowsInVal}\n")
            f.write(f"Train number correct: {numCorrect}, test number correct: {numCorrectTest}\n")
