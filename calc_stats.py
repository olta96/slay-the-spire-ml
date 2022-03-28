import json



def get_card_ids():
    with open("card_ids.json", "r") as card_ids_json_file:
        return json.loads(card_ids_json_file.read())

card_ids = get_card_ids()



def get_one_hot_encoded_data():
    with open("one_hot_encoded_data.json", "r") as one_hot_file:
        return json.loads(one_hot_file.read())

one_hot_data = get_one_hot_encoded_data()

card_appearance_counts = {}
for choice in one_hot_data:
    for card_id, value in enumerate(choice["inputs"]["available_choices"]):
        if value:
            if card_id not in card_appearance_counts:
                card_appearance_counts[card_id] = 1
            else:
                card_appearance_counts[card_id] += 1

chosen_card_counts = {}
for choice in one_hot_data:
    for card_id, value in enumerate(choice["targets"]):
        if value:
            if card_id not in chosen_card_counts:
                chosen_card_counts[card_id] = 1
            else:
                chosen_card_counts[card_id] += 1

# sort
card_appearance_counts = {k: v for k, v in sorted(card_appearance_counts.items(), key=lambda item: item[1], reverse=True)}
chosen_card_counts = {k: v for k, v in sorted(chosen_card_counts.items(), key=lambda item: item[1], reverse=True)}

toWrite = "Card appearances:"
for card_id in card_appearance_counts:
    toWrite += f"\n\t{card_ids[card_id]}: {round((card_appearance_counts[card_id] / len(one_hot_data)) * 100, 2)} %"

toWrite += "\n\nCard picked:"
for card_id in chosen_card_counts:
    toWrite += f"\n\t{card_ids[card_id]}: {round((chosen_card_counts[card_id] / len(one_hot_data)) * 100, 2)} %"

with open("stats.txt", "w") as stats_file:
    stats_file.write(toWrite)
