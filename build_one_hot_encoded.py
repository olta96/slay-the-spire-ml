import json



def get_choices_data():
    with open("choices.json", "r") as choices_json_file:
        return json.loads(choices_json_file.read())

choices_data = get_choices_data()

standard_list = []

for i in range(208):
    standard_list.append(0)

def create_one_hot_encoded_list(choices):
    one_hot_encoded_list = standard_list.copy()
    for choice in choices:
        one_hot_encoded_list[choice] = 1
    
    return one_hot_encoded_list

one_hot_encodeds = []
for run in choices_data:
    one_hot_encodeds.append(
        {
            "inputs": create_one_hot_encoded_list(run["choices"]),
            "outputs": create_one_hot_encoded_list([run["player_choice"]])
        }
    )



with open("one_hot_encoded.json", "w+") as json_file:
    json_file.write(json.dumps(one_hot_encodeds, indent=4))

