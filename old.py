import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Softmax, CrossEntropyLoss
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD
from numpy import argmax, vstack
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder




# dataset definition
class ChoicesDataset(Dataset):
    # load the dataset
    def __init__(self, choices):
        # store the inputs and outputs
        self.X = []
        self.y = []

        for choice in choices:
            self.X.append(choice["choices"])
            self.y.append(choice["player_choice"])
        
        encoder = OneHotEncoder(sparse = False)
        # transform data
        onehot = encoder.fit_transform(self.X)
        print(onehot)
 
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
        self.hidden3 = Linear(8, 3)
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
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            targets = yhat(torch.FloatTensor(10).uniform_(0, 120).long())
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
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
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat




# create the dataset
dataset = ChoicesDataset(choices)


train, test = dataset.get_splits()

# create a data loader for train and test sets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)
print(len(train_dl.dataset), len(test_dl.dataset))

model = MLP(4)

train_model(train_dl, model)

acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)

row = [ 7, 2, 3, 0 ]
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# for i, (inputs, outputs) in enumerate(train_dl):



