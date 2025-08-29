import torch, numpy, pandas, sqlite3
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tkinter import * #for the GUI

data_len = len(pandas.read_csv("data.csv").iloc[0]) #The amount of rows in the CSV file, representing the number of entries of the training data.
batch_size = (data_len // 4) + 1 #The size of the batch of each epoch. Should sacle with the size of the data; I would consider hard coding this to 5.

class ProseDataset(Dataset):
    def __init__(self, file_path, label_path, transform=None,target_transform=None):
        pandas.options.display.max_rows = 100 #Panda's default is 60
        self.data = pandas.read_csv(file_path)
        self.labels = pandas.read_csv(label_path)

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

# Initializing the NeutralNetwork class.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim = 0, end_dim = 0)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(data_len, 64), #Sends each column of the batch number's row to 64 input neurons in a linear layer.
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) #The output neurons here represent labels.
        )

    def forward(self, x):
        #x = self.flatten(x) 
        x = torch.flatten(x, start_dim = 0, end_dim = 0)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

# This try block tries to load a model to write to it
try:
    model.load_state_dict(torch.load("neural_net.pth", weights_only=True))
except:
    print("neural_net.pth not found, creating a new neural network.")

#This next block uses the tkinter library to create a GUI.
main_window = Tk(screenName=None, baseName=None, className='Tk', useTk=1)
main_window.mainloop()