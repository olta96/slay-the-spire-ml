import flask, json, torch, numpy as np
from numpy import argmax

from source_folder_path import save_model_path

from MLP import MLP

from preprocessing.CardIdentifier import CardIdentifier
from preprocessing.RelicIdentifier import RelicIdentifier
from preprocessing.OneHotEncoder import OneHotEncoder

def setup_one_hot_encoder():
    def read_config_json():
        with open("config.json") as config_file:
            return json.loads(config_file.read())

    config_options = read_config_json()

    def read_card_ids():
        with open(config_options["card_ids_json_filename"]) as card_ids_json_file:
            return json.loads(card_ids_json_file.read())

    def read_relic_ids():
        with open(config_options["relic_ids_json_filename"]) as relic_ids_json_file:
            return json.loads(relic_ids_json_file.read())

    def read_max_floor_reached():
        with open(config_options["max_floor_reached_json_filename"]) as max_floor_reached_json_file:
            return json.loads(max_floor_reached_json_file.read())["max_floor_reached"]

    card_ids = read_card_ids()
    relic_ids = read_relic_ids()
    max_floor_reached = read_max_floor_reached()

    card_identifier = CardIdentifier(card_ids)
    relic_identifier = RelicIdentifier(relic_ids)
    one_hot_encoder = OneHotEncoder(card_identifier, relic_identifier, config_options["preprocessor"]["deck_max_card_count"])

    def read_one_hot_encoded_data():
        with open("one_hot_encoded_data.json") as one_hot_encoded_data_json_file:
            return json.loads(one_hot_encoded_data_json_file.read())

    number_of_inputs = (len(read_one_hot_encoded_data()[0]["inputs"]["deck"][0]) + 1) * len(card_ids) + len(relic_ids) + max_floor_reached
    number_of_outputs = len(card_ids)

    return card_identifier, relic_identifier, one_hot_encoder, max_floor_reached, number_of_inputs, number_of_outputs






card_identifier, relic_identifier, one_hot_encoder, max_floor_reached, number_of_inputs, number_of_outputs = setup_one_hot_encoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = MLP(number_of_inputs, number_of_outputs)
model.load_state_dict(torch.load(save_model_path + "/model.pth", map_location=device))
model.eval()



def one_hot_encode_state(state):
    card_ids_len = len(card_identifier.get_card_ids())
    relic_ids_len = len(relic_identifier.get_relic_ids())

    i = 0
    while i < len(state["relics"]):
        if state["relics"][i] not in relic_identifier.get_relic_ids():
            state["relics"].pop(i)
            i -= 1
        i += 1

    choice = {
        "deck": card_identifier.identify(*state["deck"]),
        "relics": relic_identifier.identify(*state["relics"], always_return_list=True),
        "available_choices": card_identifier.identify(*state["available_choices"]),
        "floor": state["floor"],
        "player_choice": 0,
    }

    if card_ids_len != len(card_identifier.get_card_ids()):
        print("Card ids changed!")
        print("Deck", state["deck"])
        print("Choices", state["available_choices"])
    if relic_ids_len != len(relic_identifier.get_relic_ids()):
        print("Relic ids changed!")
        print("Relics", state["relics"])

    one_hot_encoded = one_hot_encoder.encode([choice], max_floor_reached)[0]["inputs"]

    flattened = one_hot_encoded["available_choices"].copy()
    for floor in one_hot_encoded["floor"]:
        flattened.append(floor)
    for relic in one_hot_encoded["relics"]:
        flattened.append(relic)
    for counts in one_hot_encoded["deck"]:
        for count in counts:
            flattened.append(count)

    return flattened



def predict(state_inputs):
    state_inputs = np.array([state_inputs], dtype=np.float32)

    state_inputs = torch.tensor(state_inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = model(state_inputs).to(device)
    probs = probs.cpu()
    probs = probs.numpy()
    np.set_printoptions(precision=4, suppress=True)
    print(probs)

    # returns the 4 indices with the highest values using numpy
    return np.argsort(probs)[0][-4:]



app = flask.Flask(__name__)

@app.route('/make_choice', methods=["POST"])
def make_choice():
    state = flask.request.get_json()
    one_hot_encoded_state = one_hot_encode_state(state)
    model_answers = predict(one_hot_encoded_state)
    model_answers_ids = []
    for i in range(len(model_answers)):
        model_answers_ids.append(card_identifier.get_card_ids()[model_answers[i]])
    print("Model answer:", model_answers_ids[3])
    print("Did not choose:", model_answers_ids[2])
    print("Did not choose:", model_answers_ids[1])
    print("Did not choose:", model_answers_ids[0])
    return flask.jsonify({"model_answer": model_answers_ids[3]})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
