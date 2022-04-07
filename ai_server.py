import flask, json, torch, numpy as np

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
    one_hot_encoder = OneHotEncoder(card_identifier, relic_identifier, config_options["preprocessor"]["deck_max_card_count"], config_options["preprocessor"]["one_hot_encode_floor"], config_options["preprocessor"]["acts"])

    def read_one_hot_encoded_data():
        with open("one_hot_encoded_data.json") as one_hot_encoded_data_json_file:
            return json.loads(one_hot_encoded_data_json_file.read())

    number_of_inputs = (len(read_one_hot_encoded_data()[0]["inputs"]["deck"][0]) + 1) * len(card_ids) + len(relic_ids) + len(config_options["preprocessor"]["acts"])
    if config_options["preprocessor"]["one_hot_encode_floor"]:
        number_of_inputs += max_floor_reached
    number_of_outputs = len(card_ids)

    return card_identifier, relic_identifier, one_hot_encoder, max_floor_reached, number_of_inputs, number_of_outputs, config_options["preprocessor"]["one_hot_encode_floor"], config_options["preprocessor"]["acts"]






card_identifier, relic_identifier, one_hot_encoder, max_floor_reached, number_of_inputs, number_of_outputs, include_floor, acts = setup_one_hot_encoder()

device = torch.device("cpu")
print("Device:", device)

model = MLP(number_of_inputs, number_of_outputs)
model.load_state_dict(torch.load(save_model_path + "/model.pth", map_location=device))
model.eval()



USE_SOFTMAX_FOR_DECK_PREDICTION = input("Use softmax for deck prediction? (y/n) ") == "y"



def one_hot_encode_state(state):
    card_ids_len = len(card_identifier.get_card_ids())
    relic_ids_len = len(relic_identifier.get_relic_ids())

    i = 0
    while i < len(state["relics"]):
        if state["relics"][i] not in relic_identifier.get_relic_ids():
            state["relics"].pop(i)
            i -= 1
        i += 1

    i = 0
    while i < len(state["deck"]):
        if state["deck"][i] not in card_identifier.get_card_ids():
            state["deck"].pop(i)
            i -= 1
        i += 1

    i = 0
    while i < len(state["available_choices"]):
        if state["available_choices"][i] not in card_identifier.get_card_ids():
            state["available_choices"].pop(i)
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

    flattened = []

    for act in one_hot_encoded["acts"]:
        flattened.append(act)
    for available_choice in one_hot_encoded["available_choices"]:
        flattened.append(available_choice)
    for relic in one_hot_encoded["relics"]:
        flattened.append(relic)
    for counts in one_hot_encoded["deck"]:
        for count in counts:
            flattened.append(count)

    if include_floor:
        for floor in one_hot_encoded["floor"]:
            flattened.append(floor)

    return one_hot_encoded, flattened


def print_probs(probs):
    probs_ids = []
    for i in range(len(probs[0])):
        probs_ids.append({
            "card_id": card_identifier.get_card_ids()[i],
            "value": probs[0][i]
        })

    # sort probs_ids by card_id
    probs_ids = sorted(probs_ids, key=lambda x: x["card_id"])

    i = 0
    j = 0
    while i < len(probs_ids):
        while j < 4:
            print("{:<20} {:>10.4f}".format(probs_ids[i]["card_id"], probs_ids[i]["value"]), end=", ")
            i += 1
            j += 1
            if i == len(probs_ids):
                break
        j = 0
        print("")

def convert_probs_to_percentages(probs):
    total = 0
    for i in range(len(probs[0])):
        total += abs(probs[0][i])
    for i in range(len(probs[0])):
        probs[0][i] = abs(probs[0][i]) / total * 100
    return probs

def predict(state_inputs, allowed_choices):
    state_inputs = np.array([state_inputs], dtype=np.float32)

    state_inputs = torch.tensor(state_inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = model(state_inputs).to(device)
    probs = probs.cpu()
    probs = probs.numpy()
    print_probs(probs)

    answers = []
    for allowed_choice in allowed_choices:
        answers.append({"card_id": card_identifier.get_card_ids()[allowed_choice], "value": probs[0][allowed_choice]})

    if USE_SOFTMAX_FOR_DECK_PREDICTION:
        for_softmaxing = []
        for answer in answers:
            for_softmaxing.append(answer["value"])
        print(for_softmaxing)
        for_softmaxing = torch.softmax(torch.tensor([for_softmaxing], dtype=torch.float32), dim=1).cpu().numpy()
        for i, answer in enumerate(answers):
            answer["value"] = for_softmaxing[0][i]

    return sorted(answers, key=lambda x: x["value"], reverse=True)



def sort_and_match_cards(cards_a, cards_b):
    sorted_a = sorted(cards_a)
    sorted_b = sorted(cards_b)

    if len(sorted_a) == len(sorted_b):
        return sorted_a == sorted_b

    for card_a in sorted_a:
        for card_b in sorted_b:
            if card_a != card_b:
                return False

    return True

def validate_cards_and_act(original_state, one_hot_encoded_state):
    original_floor = original_state["floor"]
    for i, act in enumerate(acts):
        if original_floor >= act["from_floor_inclusive"] and original_floor <= act["to_floor_inclusive"]:
            if one_hot_encoded_state["acts"][i] != 1:
                return False
        else:
            if one_hot_encoded_state["acts"][i] != 0:
                return False

    original_deck = original_state["deck"].copy()
    original_choices = original_state["available_choices"].copy()
    one_hot_deck_ids = one_hot_encoded_state["deck"]
    one_hot_choices_ids = one_hot_encoded_state["available_choices"]

    one_hot_deck = []
    one_hot_choices = []

    for i, one_hot_deck_id in enumerate(one_hot_deck_ids):
        if one_hot_deck_id == 1:
            one_hot_deck.append(card_identifier.get_card_ids()[i])
    
    for i, one_hot_choices_id in enumerate(one_hot_choices_ids):
        if one_hot_choices_id == 1:
            one_hot_choices.append(card_identifier.get_card_ids()[i])

    return sort_and_match_cards(original_deck, one_hot_deck) and sort_and_match_cards(original_choices, one_hot_choices)

app = flask.Flask(__name__)

def print_model_answers(model_answers_ids):
    print(f"\n\tModel answer: {model_answers_ids[0]['card_id']} ({'{:.4f}'.format(model_answers_ids[0]['value'])})\n")
    print(f"Did not choose: {model_answers_ids[1]['card_id']} ({'{:.4f}'.format(model_answers_ids[1]['value'])})")
    print(f"Did not choose: {model_answers_ids[2]['card_id']} ({'{:.4f}'.format(model_answers_ids[2]['value'])})")
    print(f"Did not choose: {model_answers_ids[3]['card_id']} ({'{:.4f}'.format(model_answers_ids[3]['value'])})")

@app.route('/make_choice', methods=["POST"])
def make_choice():
    state = flask.request.get_json()
    print(state)
    
    one_hot_encoded_state, flattened = one_hot_encode_state(state)
    
    allowed_choices = [0]
    for choice in state["available_choices"]:
        allowed_choices.append(card_identifier.get_card_ids().index(choice))
    model_answers_ids = predict(flattened, allowed_choices)
    
    print("State did validate" if validate_cards_and_act(state, one_hot_encoded_state) else "State did not validate")
    print_model_answers(model_answers_ids)
    
    return flask.jsonify({"model_answer": model_answers_ids[0]["card_id"]})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
