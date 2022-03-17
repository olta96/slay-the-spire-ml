import json
import numpy as np

print("started one-hot encoding")
def get_choices_data():
    with open("choices.json", "r") as choices_json_file:
        return json.loads(choices_json_file.read())

choices_data = get_choices_data()

standard_list = []

for i in range(208):
    standard_list.append(0)

def create_one_hot_encoded_list(choices):
    one_hot_encoded_list = standard_list.copy()
    for choice in choices:
        one_hot_encoded_list[choice] = 1
    
    return one_hot_encoded_list

one_hot_encoded_data = []
for run in choices_data:
    one_hot_encoded_data.append(
        {
            "inputs": create_one_hot_encoded_list(run["choices"]),
            "outputs": create_one_hot_encoded_list([run["player_choice"]])
        }
    )

print("one-hot encoding complete")









import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Softmax, CrossEntropyLoss
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD
from numpy import argmax, dtype, vstack
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



# dataset definition
class ChoicesDataset(Dataset):
    # load the dataset
    def __init__(self, one_hot_encoded_data):
        # store the inputs and outputs
        self.X = []
        self.y = []

        for choice in one_hot_encoded_data:
            self.X.append(choice["inputs"])
            self.y.append(choice["outputs"])
        
        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(device)

 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self):
        test_size = round(0.33 * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])




# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 208)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X




# train the model
def train_model(train_dl, model: MLP):
    # define the optimization
    max_epochs = 100
    ep_log_interval = 10
    loss_func = torch.nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    

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
def evaluate_model(model: MLP, train_ds):
    return accuracy(model, train_ds)


    # predictions, actuals = list(), list()

    # for i, (inputs, targets) in enumerate(test_dl):
    #     # evaluate the model on the test set
    #     yhat = model(inputs)
    #     # retrieve numpy array
    #     yhat = yhat.detach().numpy()
    #     actual = targets.numpy()
    #     # convert to class labels
    #     yhat = argmax(yhat, axis=1)
    #     # reshape for stacking
    #     actual = actual.reshape((len(actual), 1))
    #     yhat = yhat.reshape((len(yhat), 1))
    #     # store
    #     predictions.append(yhat)
    #     actuals.append(actual)

    # predictions, actuals = vstack(predictions), vstack(actuals)
    # # calculate accuracy
    # acc = accuracy_score(actuals, predictions)
    # return acc




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
model = MLP(208).to(device)

print("Started training model")
train_model(train_dl, model)
print("Training complete")

acc = accuracy(model, test_dl)
print('Accuracy: %.3f' % acc)
row = [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i, val in enumerate(row):
    if val == 1:
        print(i)

row = np.array([row], dtype=np.float32)

row = torch.tensor(row, dtype=torch.float32).to(device)

predict(row, model)

