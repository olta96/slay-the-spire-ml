import json

from data_source_path import data_source_path



def read_json_file():
    with open(data_source_path + "/2020-11-01-00-07#1067.json", "r") as json_file:
        json_contents = json_file.read()
        return json.loads(json_contents)
    
runs = read_json_file()
print(f"Loaded: {len(runs)} runs.")

result_runs = []
for run in runs:
    event = run["event"]
    if event["character_chosen"] == "IRONCLAD"\
    and event["ascension_level"] >= 10\
    and event["floor_reached"] >= 44:
        result_runs.append(run)

with open("out.json", "w+") as json_file:
    json_file.write(json.dumps(result_runs, indent=4))

print(f"Exported: {len(result_runs)} runs.")

