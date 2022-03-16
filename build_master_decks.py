import json


def get_filtered_data():
    with open("filtered.json", "r") as filtered_json_file:
        return json.loads(filtered_json_file.read())

filtered_data = get_filtered_data()

master_decks = []
for run in filtered_data:
    master_decks.append({"master_deck": run["master_deck"], "reached_floor_44": run["floor_reached"] >= 44})


with open("master_decks.json", "w+") as json_file:
    json_file.write(json.dumps(master_decks, indent=4))

