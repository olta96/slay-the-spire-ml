import json

from data_source_path import data_source_path



def read_json_file():
    with open(data_source_path + "/2020-11-01-00-07#1067.json", "r") as json_file:
        json_contents = json_file.read()
        return json.loads(json_contents)
    
runs = read_json_file()
print(f"Loaded: {len(runs)} runs.")

result_runs = []
_id = -1
for run in runs:
    event = run["event"]
    if event["character_chosen"] == "IRONCLAD"\
    and event["ascension_level"] >= 10\
    and event["floor_reached"] >= 44\
    and not event["is_endless"]:
        _id += 1
        new_run = {}
        new_run["id"] = _id
        new_run["victory"] = event["victory"]
        new_run["ascension_level"] = event["ascension_level"]
        new_run["floor_reached"] = event["floor_reached"]
        new_run["card_choices"] = event["card_choices"]
        new_run["event_choices"] = []
        for event_choice in event["event_choices"]:
            for value in ["cards_optained", "cards_transformed", "cards_removed", "copied", "floor"]:
                if value in event_choice.keys():
                    new_run["event_choices"].append(event_choice)
                    break
        new_run["campfire_choices"] = []
        for campfire_choice in event["campfire_choices"]:
            if campfire_choice["key"] == "SMITH":
                new_run["campfire_choices"].append(campfire_choice)
        result_runs.append(new_run)

with open("out.json", "w+") as json_file:
    json_file.write(json.dumps(result_runs, indent=4))

print(f"Exported: {len(result_runs)} runs.")

