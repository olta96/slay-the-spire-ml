from preprocessing.Preprocesser import Preprocesser
import json

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim import Adam, AdamW
from numpy import argmax, vstack
from sklearn.metrics import accuracy_score
from source_folder_path import save_model_path


def read_config_json():
    with open("config.json") as config_file:
        return json.loads(config_file.read())

config_options = read_config_json()

ONE_HOT_ENCODED_JSON_FILENAME = "one_hot_encoded_data.json"
CARD_IDS_JSON_FILENAME = "card_ids.json"


if input(f"Run preprocesser (y/n): ") == "y":
    preprocesser = Preprocesser(config_options["preprocessor"], ONE_HOT_ENCODED_JSON_FILENAME, CARD_IDS_JSON_FILENAME)
    preprocesser.start()
    one_hot_encoded_data = preprocesser.get_one_hot_encoded_data()
    card_ids = preprocesser.get_card_ids()
else:
    with open(ONE_HOT_ENCODED_JSON_FILENAME, "r") as json_file:
        one_hot_encoded_data = json.loads(json_file.read())
    with open(CARD_IDS_JSON_FILENAME, "r") as json_file:
        card_ids = json.loads(json_file.read())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

number_of_inputs = (len(one_hot_encoded_data[0]["inputs"]["deck"][0]) + 1) * len(card_ids)
print("Number of input nodes:", number_of_inputs)

# For predicting
test_row = None

# dataset definition
class ChoicesDataset(Dataset):
    # load the dataset
    def __init__(self, one_hot_encoded_data):
        global test_row
        # store the inputs and outputs
        self.X = []
        self.y = []

        for choice in one_hot_encoded_data:
            to_append = choice["inputs"]["available_choices"].copy()
            for counts in choice["inputs"]["deck"]:
                for count in counts:
                    to_append.append(count)
            
            self.X.append(to_append)
            self.y.append(choice["targets"])

        test_row = self.X[0].copy()

        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(device)

 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self):
        test_size = round(0.1 * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])




# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()

        self.hid1 = torch.nn.Linear(n_inputs, int(n_inputs / 2))
        self.hid2 = torch.nn.Linear(int(n_inputs / 2), int(n_inputs / 2))
        self.oupt = torch.nn.Linear(int(n_inputs / 2), len(card_ids))

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

 
    # forward propagate input
    def forward(self, X):
        z = torch.tanh(self.hid1(X))
        z = torch.tanh(self.hid2(z))
        # No softmax, happens in CrossEntropyLoss
        z = self.oupt(z)
        return z




# train the model
def train_model(train_dl, model, test_dl):
    max_epochs = config_options["max_epochs"]
    epoch_log_interval = config_options["epoch_log_interval"]
    loss_func = torch.nn.CrossEntropyLoss()
    
    # define the optimization
    lr = config_options["learning_rate"]
    if config_options["optimizer"] == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif config_options["optimizer"] == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise f"Unknown optimizer function: {config_options['optimizer']}"


    # enumerate epochs
    for epoch in range(max_epochs):
        # enumerate mini batches
        epoch_loss = 0
        
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = loss_func(yhat, targets)
            epoch_loss += loss.item()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

        if epoch % epoch_log_interval == 0:
            acc = accuracy(model, test_dl)
            acc_percentage = round(acc * 100, 2)
            print("epoch = %4d   accuracy = %0.2f %%   loss = %0.4f" % (epoch, acc_percentage, epoch_loss))

def accuracy(model, ldr):
    # assumes model.eval()
    n_correct = 0; n_wrong = 0
    # using loader avoids resize() issues
    for _, (X, Y) in enumerate(ldr):
        with torch.no_grad():
            oupt = model(X)  # probs form    
        if torch.argmax(Y) == torch.argmax(oupt):
            n_correct += 1
        else:
            n_wrong += 1
    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc

# evaluate the model
def evaluate_model(model: MLP):
    predictions, actuals = list(), list()

    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc




# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    with torch.no_grad():
        probs = model(row).to(device)  # values sum to 1.0
    probs = probs.cpu()
    probs = probs.numpy()  # numpy vector prints better
    np.set_printoptions(precision=4, suppress=True)
    print(probs)
    print('(class=%f)' % (argmax(probs)))




# create the dataset
print("Creating dataset")
dataset = ChoicesDataset(one_hot_encoded_data)

print("Splitting dataset")
train, test = dataset.get_splits()


print("DataLoader loading dataset")
# create a data loader for train and test sets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1, shuffle=False)

print("Creating model")
model = MLP(number_of_inputs).to(device)

print("Started training model")
train_model(train_dl, model, test_dl)
print("Training complete")

acc = accuracy(model, test_dl)
print(f"Accuracy: {round(acc * 100, 2)} %")

print("Test row available choices:")
for i, val in enumerate(test_row):
    if i >= len(card_ids):
        break
    if val == 1:
        print(f"\t{i}: {card_ids[i]}")

test_row = np.array([test_row], dtype=np.float32)

test_row = torch.tensor(test_row, dtype=torch.float32).to(device)

predict(test_row, model)

torch.save(model.state_dict(), save_model_path + "/model.pth")
print(f"Saved PyTorch Model State to {save_model_path}\model.pth")
