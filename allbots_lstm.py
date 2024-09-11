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

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, batch_first=True)
        self.linear = nn.Linear(50, 3)

    def forward(self, x):
        # print(x)
        x, _ = self.lstm(x.to(torch.float32))
        x = self.linear(x[:, -1, :])  # Get only the last output
        return x

def load_data_from_folder(dir: str, train_split: float = 0.8):
    """Load data from all npy files in a folder and convert it to 2 tensors: 1 training, 1 testing."""
    paths = [os.path.join(dir, p) for p in os.listdir(dir) if p.endswith(".npy")]
    
    # Load data into single np array
    data = [np.load(filepath)[:20] for filepath in paths]
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
    
    def getItem(self, index, subbatch_number: int, subbatch_size: int = 100):
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
    """Iterate over a dataset."""
    procNums = list(range(8))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for batchNum in tqdm(range(len(dataset))):
            # print(batchNum, '/', len(dataset))
            for X_batch, y_batch in executor.map(dataset.getItem, [batchNum]*8, procNums):
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
    trainData, testData = load_data_from_folder("Greenberg_AllBots_Data/")

    model = LSTM()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    dataset = WindowDataset(trainData)
    valSet = WindowDataset(testData)

    n_epochs = 100
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        numCorrect = 0
        dataset.shuffleData()
        model.train()
        # for batchNum in range(len(dataset)):
        #     X, y = dataset[batchNum]
        #     for i in range(8):
        #         X_batch = X[100*i:100*(i+1)]
        #         y_batch = y[100*i:100*(i+1)]
                # y_pred = model(X_batch)
                # loss = loss_fn(y_pred, y_batch)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
        for X_batch, y_batch in iterateOverDataset(dataset):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            numCorrect += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().sum()
        
        # Validation done every 3 epochs
        if epoch % 3 != 0:
            print(f"Epoch {epoch}: train accuracy {100 * numCorrect / (len(dataset) * 800)}%")
            continue
        model.eval()
        numCorrectTest = 0
        with torch.no_grad():
            for X_batch, y_batch in iterateOverDataset(valSet):
                y_pred = model(X_batch)
                numCorrectTest += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().sum()
        print(f"Epoch {epoch}: train accuracy {100 * numCorrect / (len(dataset) * 800)}%, test accuracy {100 * numCorrectTest / (len(valSet) * 800)}%")

        # Save model
        directoryPath = f"D:/cs486/lstm_models_allbots_new/{epoch}/"
        os.makedirs(directoryPath, exist_ok=False)
        torch.save(model.state_dict(), os.path.join(directoryPath, "model_stateDict.pt"))
        torch.save(model, os.path.join(directoryPath, "model.pt"))

        with open(os.path.join(directoryPath, "accuracy.txt"), "w") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"Train accuracy {100 * numCorrect / (len(dataset) * 800)}%, test accuracy {100 * numCorrectTest / (len(valSet) * 800)}%\n")
            f.write(f"Train dataset size: {len(dataset) * 800}, test dataset size: {len(valSet) * 800}\n")
            f.write(f"Train number correct: {numCorrect}, test number correct: {numCorrectTest}\n")
