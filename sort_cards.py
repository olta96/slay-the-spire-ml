import json


starting_cards = [
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Defend_R",
    "Defend_R",
    "Defend_R",
    "Defend_R",
    "Bash",
    "AscendersBane",
]


def get_filtered_data():
    with open("filtered.json", "r") as filtered_json_file:
        return json.loads(filtered_json_file.read())

filtered_data = get_filtered_data()


def modify_cards_by_card_choice(card_choice, cards, mod_events):
    if card_choice["picked"] != "SKIP":
        cards.append(card_choice["picked"])
        mod_events.append("picked: " + card_choice["picked"])


def modify_cards_by_event(event, cards, mod_events):
    if "cards_removed" in event:
        for card_removed in event["cards_removed"]: # fulfix: "Rupture" problemet, saknar första event
            if card_removed in cards:
                cards.remove(card_removed)
                mod_events.append("removed: " + card_removed)
    
    if "cards_transformed" in event:
        for card_transformed in event["cards_transformed"]:
            cards.remove(card_transformed)
            mod_events.append("transformed (removed): " + card_transformed)

    if "cards_obtained" in event:
        for card_obtained in event["cards_obtained"]:
            cards.append(card_obtained)
            mod_events.append("obtained: " + card_obtained)

    if "cards_upgraded" in event:
        for card_upgraded in event["cards_upgraded"]:
            # fulfix: lös "+2"
            # splitted = card_upgraded.split("+")
            # if len(splitted):

            cards.remove(card_upgraded)
            cards.append(card_upgraded + "+1")
            mod_events.append("upgraded: " + card_upgraded)


def modify_cards_by_campfire(campfire_choice, cards, mod_events):
    cards.remove(campfire_choice["data"])
    cards.append(campfire_choice["data"] + "+1")
    mod_events.append("upgraded: " + campfire_choice["data"])


def modify_cards_by_purchase(card_purchased, cards, mod_events):
    cards.append(card_purchased["card"])
    mod_events.append("purchased: " + card_purchased["card"])


def sort_cards_by_floor(run):
    floor_hands = []
    cards = starting_cards.copy()
    for floor in range(0, 60):
        mod_events = []
        for card_choice in run["card_choices"]:
            if card_choice["floor"] > floor:
                break
            if card_choice["floor"] == floor:
                modify_cards_by_card_choice(card_choice, cards, mod_events)
        
        for event_choice in run["event_choices"]:
            if event_choice["floor"] > floor:
                break
            if event_choice["floor"] == floor:
                modify_cards_by_event(event_choice, cards, mod_events)

        for campfire_choice in run["campfire_choices"]:
            if campfire_choice["floor"] > floor:
                break
            if campfire_choice["floor"] == floor:
                modify_cards_by_campfire(campfire_choice, cards, mod_events)

        for card_purchased in run["cards_purchased"]:
            if card_purchased["floor"] > floor:
                break
            if card_purchased["floor"] == floor:
                modify_cards_by_purchase(card_purchased, cards, mod_events)

        if not len(mod_events):
            mod_events = "no change"
        
        floor_hands.append({
            "event": mod_events,
            "floor": floor,
            "cards": cards.copy(),
        })
    
    return floor_hands



run_floor_hands = []
for run in filtered_data:
    run_floor_hands.append(sort_cards_by_floor(run))
    if run["master_deck"].sort() == run_floor_hands[-1][-1]["cards"].sort():
        print("last hand matches master_deck")
    else:
        print("last hand does not match master_deck")

with open("cards_per_floor.json", "w+") as json_file:
    json_file.write(json.dumps(run_floor_hands, indent=4))
