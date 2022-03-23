from preprocessing.Preprocesser import Preprocesser
import json

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Softmax, CrossEntropyLoss
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD, Adam, AdamW
from numpy import argmax, dtype, vstack
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from source_folder_path import save_model_path

# print("Loading card_ids")
# def get_card_ids():
#     with open("card_ids.json", "r") as card_ids_json_file:
#         return json.loads(card_ids_json_file.read())

# card_ids = get_card_ids()

# print("started one-hot encoding")
# def get_choices_data():
#     with open("choices.json", "r") as choices_json_file:
#         return json.loads(choices_json_file.read())

# choices_data = get_choices_data()

# standard_list = []

# for i in range(len(card_ids)):
#     standard_list.append(0)

# def create_one_hot_encoded_list(choices):
#     one_hot_encoded_list = standard_list.copy()
#     for choice in choices:
#         one_hot_encoded_list[choice] = 1
    
#     return one_hot_encoded_list

# one_hot_encoded_data = []
# for run in choices_data:
#     one_hot_encoded_data.append(
#         {
#             "inputs": create_one_hot_encoded_list(run["choices"]),
#             "outputs": create_one_hot_encoded_list([run["player_choice"]])
#         }
#     )

# print("one-hot encoding complete")

ONE_HOT_ENCODED_JSON_FILENAME = "one_hot_encoded_data.json"
CARD_IDS_JSON_FILENAME = "card_ids.json"

if input(f"Run preprocesser (y/n): ") == "y":
    preprocesser = Preprocesser(ONE_HOT_ENCODED_JSON_FILENAME, CARD_IDS_JSON_FILENAME)
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

row = None
# dataset definition
class ChoicesDataset(Dataset):
    # load the dataset
    def __init__(self, one_hot_encoded_data):
        global row
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

        row = self.X[0].copy()

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

        # # input to first hidden layer
        # self.hidden1 = Linear(n_inputs, 10)
        # xavier_uniform_(self.hidden1.weight)
        # self.act1 = ReLU()
        # # second hidden layer
        # self.hidden2 = Linear(10, 8)
        # xavier_uniform_(self.hidden2.weight)
        # self.act2 = ReLU()
        # # third hidden layer and output
        # self.hidden3 = Linear(8, n_inputs)
        # xavier_uniform_(self.hidden3.weight)
        # self.act3 = Softmax(dim=1)
 
    # forward propagate input
    def forward(self, X):
        z = torch.tanh(self.hid1(X))
        z = torch.tanh(self.hid2(z))
        z = self.oupt(z) # NOTE?
        return z

        # # input to first hidden layer
        # X = self.hidden1(X)
        # X = self.act1(X)
        # # second hidden layer
        # X = self.hidden2(X)
        # X = self.act2(X)
        # # output layer
        # X = self.hidden3(X)
        # X = self.act3(X)
        # return X




# train the model
def train_model(train_dl, model: MLP):
    # define the optimization
    max_epochs = 2
    ep_log_interval = 1
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.002)


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

        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

    print("Done ")

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
model = MLP(len(card_ids) * 7).to(device)

print("Started training model")
train_model(train_dl, model)
print("Training complete")

acc = accuracy(model, test_dl)
print(f"Accuracy: {round(acc * 100, 2)} %")

choices = []
for i in range(1, len(row)):
    if i == len(row) - 1:
        break
    temp = row[i]
    row[i] = row[i + 1]
    row[i + 1] = temp
for i, val in enumerate(row):
    if val == 1:
        choices.append(i)
        print(i)

row = np.array([row], dtype=np.float32)

row = torch.tensor(row, dtype=torch.float32).to(device)

predict(row, model)

torch.save(model.state_dict(), save_model_path + "/model.pth")
print(f"Saved PyTorch Model State to {save_model_path}\model.pth")
