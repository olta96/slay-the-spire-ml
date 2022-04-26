from preprocessing.Preprocesser import Preprocesser
from Plotter import Plotter
import json
import copy

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim import Adam, AdamW
from numpy import argmax, vstack
from sklearn.metrics import accuracy_score
from source_folder_path import save_model_path
from MLP import MLP


def read_config_json():
    with open("config.json") as config_file:
        return json.loads(config_file.read())

config_options = read_config_json()

ONE_HOT_ENCODED_JSON_FILENAME = "one_hot_encoded_data.json"
CARD_IDS_JSON_FILENAME = config_options["card_ids_json_filename"]
RELIC_IDS_JSON_FILENAME = config_options["relic_ids_json_filename"]
MAX_FLOOR_REACHED_JSON_FILENAME = config_options["max_floor_reached_json_filename"]
SHOULD_INCLUDE_RELICS = config_options["include_relics"]
SHOULD_INCLUDE_FLOOR = config_options["preprocessor"]["one_hot_encode_floor"]
DATASET_SPLIT_TRAIN_SIZE = config_options["dataset_split_train_size"]


if input(f"Run preprocesser (y/n): ") == "y":
    preprocesser = Preprocesser(config_options["preprocessor"], ONE_HOT_ENCODED_JSON_FILENAME, CARD_IDS_JSON_FILENAME, RELIC_IDS_JSON_FILENAME, MAX_FLOOR_REACHED_JSON_FILENAME)
    preprocesser.start()
    one_hot_encoded_data = preprocesser.get_one_hot_encoded_data()
    card_ids = preprocesser.get_card_ids()
    relic_ids = preprocesser.get_relic_ids()
    max_floor_reached = preprocesser.get_max_floor_reached()
else:
    with open(ONE_HOT_ENCODED_JSON_FILENAME, "r") as json_file:
        one_hot_encoded_data = json.loads(json_file.read())
    with open(CARD_IDS_JSON_FILENAME, "r") as json_file:
        card_ids = json.loads(json_file.read())
    with open(RELIC_IDS_JSON_FILENAME, "r") as json_file:
        relic_ids = json.loads(json_file.read())
    with open(MAX_FLOOR_REACHED_JSON_FILENAME, "r") as json_file:
        max_floor_reached = json.loads(json_file.read())["max_floor_reached"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



number_of_inputs = len(config_options["preprocessor"]["acts"]) + (len(one_hot_encoded_data[0]["inputs"]["deck"][0]) + 1) * len(card_ids)
if SHOULD_INCLUDE_RELICS:
    number_of_inputs += len(relic_ids)
if SHOULD_INCLUDE_FLOOR:
    number_of_inputs += max_floor_reached

number_of_outputs = len(card_ids)

print("Number of input nodes:", number_of_inputs)
print("Number of output nodes:", number_of_outputs)



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
            inputs_flattened = []
            for act in choice["inputs"]["acts"]:
                inputs_flattened.append(act)
            for available_choice in choice["inputs"]["available_choices"]:
                inputs_flattened.append(available_choice)
            for counts in choice["inputs"]["deck"]:
                for count in counts:
                    inputs_flattened.append(count)

            if SHOULD_INCLUDE_RELICS:
                for relic in choice["inputs"]["relics"]:
                    inputs_flattened.append(relic)

            if SHOULD_INCLUDE_FLOOR:
                for floor in choice["inputs"]["floor"]:
                    inputs_flattened.append(floor)
            
            self.X.append(inputs_flattened)
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

    def group_by_acts(self):
        groups = []
        for _act in config_options["preprocessor"]["acts"]:
            groups.append([])
        
        for i, (inputs, _) in enumerate(self):
            for j in range(len(config_options["preprocessor"]["acts"])):
                if inputs[j] == 1:
                    groups[j].append(self[i])

        return groups

    def balance_acts(self, act_groups):
        size_of_smallets_act_group = len(act_groups[0])
        for act_group in act_groups:
            if len(act_group) < size_of_smallets_act_group:
                size_of_smallets_act_group = len(act_group)

        for i in range(len(act_groups)):
            act_groups[i] = act_groups[i][:size_of_smallets_act_group]

        return act_groups

    def get_splits(self, random=False):
        if random:
            test_size = round((1 - DATASET_SPLIT_TRAIN_SIZE) * len(self.X))
            train_size = len(self.X) - test_size
            return random_split(self, [train_size, test_size])
        else:
            act_groups = self.group_by_acts()
            act_groups.pop() # remove the last act since it is very small and rare
            act_groups = self.balance_acts(act_groups)
            train_set = []
            test_set = []
            for act_group in act_groups:
                test_size = round((1 - DATASET_SPLIT_TRAIN_SIZE) * len(act_group))
                train_size = len(act_group) - test_size
                train_set.extend(act_group[:train_size])
                test_set.extend(act_group[train_size:])

            return train_set, test_set





# train the model
def train_model(train_dl, model, test_dl):
    most_accurate_model = None
    accuracy_of_most_accurate_model = 0

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


    plotter = Plotter(f"strict_mode_decks: {config_options['preprocessor']['strict_mode_decks']} strict_mode_relics: {config_options['preprocessor']['strict_mode_relics']}")


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
            if acc > accuracy_of_most_accurate_model:
                most_accurate_model = copy.deepcopy(model)
                accuracy_of_most_accurate_model = acc
            plotter.push_epoch(epoch, epoch_loss, acc_percentage)
    
    plotter.save()
    return most_accurate_model, accuracy_of_most_accurate_model

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
train_dl = DataLoader(train, batch_size=1028, shuffle=True)
test_dl = DataLoader(test, batch_size=1, shuffle=False)
train_dl_for_accuracy = DataLoader(train, batch_size=1, shuffle=False)

print("Creating model")
model = MLP(number_of_inputs, number_of_outputs).to(device)

print("Started training model")
most_accurate_model, most_accurate_model_acc = train_model(train_dl, model, test_dl)
print("Training complete")

acc = accuracy(model, test_dl)
print(f"Current Model Accuracy for test dataset: {round(acc * 100, 2)} %")
acc = accuracy(model, train_dl_for_accuracy)
print(f"Current Model Accuracy for train dataset: {round(acc * 100, 2)} %")
print(f"Most Accurate Model Accuracy: {round(most_accurate_model_acc * 100, 2)} %")

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
