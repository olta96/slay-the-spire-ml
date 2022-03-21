import json
from reference_list import ironclad_cards, colorless_cards

choosable_cards = ironclad_cards + colorless_cards + ["SKIP"]


def get_filtered_data():
    with open("filtered.json", "r") as filtered_json_file:
        return json.loads(filtered_json_file.read())

filtered_data = get_filtered_data()

card_ids = ["SKIP"]

def identify_choice(card_name):
    if card_name in card_ids:
        return card_ids.index(card_name)
    
    card_ids.append(card_name)
    return len(card_ids) - 1

def build_choices():
    choices = []

    for run in filtered_data:
        if run["floor_reached"] < 44:
            continue
        for card_choice in run["card_choices"]:
            if len(card_choice["not_picked"]) < 2 or len(card_choice["not_picked"]) > 3:
                continue
            if len(card_choice["not_picked"]) == 3 and card_choice["picked"] != "SKIP":
                continue

            if card_choice["picked"] not in choosable_cards:
                continue

            should_continue = False
            for not_picked in card_choice["not_picked"]:
                if not_picked not in choosable_cards:
                    should_continue = True
                    break

            if should_continue:
                continue

            new_choice = { "choices": [], "player_choice": identify_choice(card_choice["picked"]) }
            if len(card_choice["not_picked"]) == 2:
                new_choice["choices"].append(identify_choice(card_choice["not_picked"][0]))
                new_choice["choices"].append(identify_choice(card_choice["not_picked"][1]))
                new_choice["choices"].append(identify_choice(card_choice["picked"]))
            else:
                new_choice["choices"].append(identify_choice(card_choice["not_picked"][0]))
                new_choice["choices"].append(identify_choice(card_choice["not_picked"][1]))
                new_choice["choices"].append(identify_choice(card_choice["not_picked"][2]))

            if 0 in new_choice["choices"]:
                continue

            new_choice["choices"].append(identify_choice("SKIP"))

            choices.append(new_choice)
    
    return choices


choices = build_choices()





for i, card_id in enumerate(card_ids):
    print(i, card_id)

with open("choices.json", "w+") as json_file:
    json_file.write(json.dumps(choices, indent=4))

with open("card_ids.json", "w+") as json_file:
    json_file.write(json.dumps(card_ids, indent=4))
