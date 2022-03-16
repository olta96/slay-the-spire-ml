import json

from data_source_path import data_source_path
from reference_list import ironclad_cards, colorless_cards


sample_data_json_file = "2020-11-01-00-37#1158.json"


def read_json_file(json_file_name):
    json_data_file_path = data_source_path + "/" + json_file_name
    with open(json_data_file_path, "r") as json_file:
        json_contents = json_file.read()
        return json.loads(json_contents)


runs = read_json_file(sample_data_json_file)

print(f"Loaded: {len(runs)} runs.")


result_runs = []


def event_matches_filters(event):
    return event["character_chosen"] == "IRONCLAD"\
    and event["ascension_level"] >= 10\
    and not event["is_endless"]\
    and event["is_ascension_mode"]\
    and not event["chose_seed"]


def append_event_choice(event_choices, event_choice):
    if any(key in event_choice.keys() for key in ["cards_obtained", "cards_removed", "cards_upgraded"]):
        new_event_choice = {}
        for key in ["cards_obtained", "cards_transformed", "cards_removed", "cards_upgraded", "floor"]:
            if key in event_choice.keys():
                new_event_choice[key] = event_choice[key]
        event_choices.append(new_event_choice)


def create_run(event):
    run = {}
    run["id"] = _id
    run["victory"] = event["victory"]
    run["ascension_level"] = event["ascension_level"]
    run["floor_reached"] = event["floor_reached"]
    run["card_choices"] = event["card_choices"]
    run["master_deck"] = event["master_deck"]
    run["play_id"] = event["play_id"]

    run["cards_purchased"] = []
    for i in range(len(event["items_purchased"])):
        for purchasable_card in ironclad_cards + colorless_cards:
            if event["items_purchased"][i].startswith(purchasable_card):
                run["cards_purchased"].append({
                    "card": event["items_purchased"][i],
                    "floor": event["item_purchase_floors"][i],
                })
                break

    run["event_choices"] = []
    for event_choice in event["event_choices"]:
        append_event_choice(run["event_choices"], event_choice)

    run["campfire_choices"] = []
    for campfire_choice in event["campfire_choices"]:
        if campfire_choice["key"] == "SMITH":
            run["campfire_choices"].append(campfire_choice)
    
    return run


_id = -1
for run in runs:
    event = run["event"]
    if event_matches_filters(event):
        _id += 1
        new_run = create_run(event)
        result_runs.append(new_run)


with open("filtered.json", "w+") as json_file:
    json_file.write(json.dumps(result_runs, indent=4))


print(f"Exported: {len(result_runs)} runs.")
