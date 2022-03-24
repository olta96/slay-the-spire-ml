import json



def get_card_ids():
    with open("card_ids.json", "r") as card_ids_json_file:
        return json.loads(card_ids_json_file.read())

card_ids = get_card_ids()



def get_choices_data():
    with open("choices.json", "r") as choices_json_file:
        return json.loads(choices_json_file.read())

choices_data = get_choices_data()

card_counts = {}
for choices in choices_data:
    for choice in choices["choices"]:
        if choice not in card_counts:
            card_counts[choice] = 0

        card_counts[choice] += 1

chosen_card_counts = {}
for choices in choices_data:
    player_choice = choices["player_choice"]
    if player_choice not in chosen_card_counts:
        chosen_card_counts[player_choice] = 0

    chosen_card_counts[player_choice] += 1

# sort
card_counts = {k: v for k, v in sorted(card_counts.items(), key=lambda item: item[1], reverse=True)}
chosen_card_counts = {k: v for k, v in sorted(chosen_card_counts.items(), key=lambda item: item[1], reverse=True)}

toWrite = "Card appearances:"

for card_id in card_counts:
    toWrite += f"\n\t{card_ids[card_id]}: {round((card_counts[card_id] / len(choices_data)) * 100, 2)} %"

toWrite += "\nCard picked:"
for card_id in chosen_card_counts:
    toWrite += f"\n\t{card_ids[card_id]}: {round((chosen_card_counts[card_id] / len(choices_data)) * 100, 2)} %"

with open("stats.txt", "w") as stats_file:
    stats_file.write(toWrite)
