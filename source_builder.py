import json, os

from source_folder_path import data_source_path
from reference_list import ironclad_cards, colorless_cards

FILE_CAP = 150
source_paths = []
print("Started loading json files, cap:", FILE_CAP)
for i, file in enumerate(os.listdir(data_source_path)):
    if i % 10 == 0:
        print("Loaded", i, "files")
    source_paths.append(file)
    if i == FILE_CAP:
        break


def read_json_file(runs, json_file_names):
    print("Started reading json files")
    for i, json_file_name in enumerate(json_file_names):
        if i % 10 == 0:
            print(i, "files read")
        json_data_file_path = data_source_path + "/" + json_file_name
        with open(json_data_file_path, "r") as json_file:
            json_contents = json_file.read()
            runs.append(json.loads(json_contents))


loaded_files = []
read_json_file(loaded_files, source_paths)

print(f"Loaded: {len(loaded_files)} files.")


result_runs = []


def event_matches_filters(event):
    return event["character_chosen"] == "IRONCLAD"\
    and event["ascension_level"] >= 10\
    and event["floor_reached"] >= 44\
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

skipped_item_purchased = 0
def create_run(event):
    global skipped_item_purchased
    run = {}
    run["play_id"] = event["play_id"]
    run["victory"] = event["victory"]
    run["ascension_level"] = event["ascension_level"]
    run["floor_reached"] = event["floor_reached"]
    run["card_choices"] = event["card_choices"]
    run["master_deck"] = event["master_deck"]

    run["cards_purchased"] = []
    if len(event["items_purchased"]) == len(event["item_purchase_floors"]):
        for i in range(len(event["items_purchased"])):
            for purchasable_card in ironclad_cards + colorless_cards:
                if event["items_purchased"][i].startswith(purchasable_card):
                    run["cards_purchased"].append({
                        "card": event["items_purchased"][i],
                        "floor": event["item_purchase_floors"][i],
                    })
                    break
    else:
        skipped_item_purchased += 1
        return None

    run["event_choices"] = []
    for event_choice in event["event_choices"]:
        append_event_choice(run["event_choices"], event_choice)

    run["campfire_choices"] = []
    for campfire_choice in event["campfire_choices"]:
        if campfire_choice["key"] == "SMITH":
            run["campfire_choices"].append(campfire_choice)
    
    return run


print("Started processing runs")
for runs in loaded_files:
    for run in runs:
        event = run["event"]
        if event_matches_filters(event):
            new_run = create_run(event)
            if new_run is not None:
                result_runs.append(new_run)

with open("filtered.json", "w+") as json_file:
    json_file.write(json.dumps(result_runs, indent=4))


print(f"Exported: {len(result_runs)} runs.")
print(f"Skipped runs: {skipped_item_purchased}")
