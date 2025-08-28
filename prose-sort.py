import torch, numpy, pandas, sqlite3
from torch import nn
from torch.utils.data import Dataset, DataLoader

data_len = len(pd.read_csv("data.csv").iloc[0]) #The amount of rows in the CSV file, representing the number of entries of the training data.
batch_size = (data_len // 4) + 1 #The size of the batch of each epoch. Should sacle with the size of the data; I would consider hard coding this to 5.

class ProseDataset(Dataset):
    def __init__(self, file_path, label_path, transform=None,target_transform=None):
        pd.options.display.max_rows = 100 #Panda's default is 60
        self.data = pd.read_csv(file_path)
        self.labels = pd.read_csv(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data.iloc[idx].astype(numpy.float32).to_numpy()), self.labels.iloc[idx, 0] #Returns an array with one data entry and an integer representing its label.

training_data = ProseDataset(
    file_path="data.csv",
    label_path = "data_labels.csv"
)

test_data = ProseDataset(
    file_path="test.csv",
    label_path = "test_labels.csv"
)

# Data loaders prototyped from PyTorch.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"