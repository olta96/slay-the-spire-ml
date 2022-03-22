MAX_FLOOR = 60

class DeckBuilder:

    def __init__(self):
        self.count_master_decks = 0
        self.count_master_decks_failed = 0
        self.prints = 0
        self.print_once = True

        self.starting_deck = [
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

    def build(self, run):
        decks = []
        cards = self.starting_deck.copy()
        for floor in range(MAX_FLOOR):
            self.prints += 1
            if self.prints == 1:
                print(run)

            for card_choice in run["card_choices"]:
                if card_choice["floor"] > floor:
                    break
                if card_choice["floor"] == floor:
                    self.modify_cards_by_card_choice(card_choice, cards)
            
            for event_choice in run["event_choices"]:
                if event_choice["floor"] > floor:
                    break
                if event_choice["floor"] == floor:
                    self.modify_cards_by_event(event_choice, cards, decks)

            for campfire_choice in run["campfire_choices"]:
                if campfire_choice["floor"] > floor:
                    break
                if campfire_choice["floor"] == floor:
                    self.modify_cards_by_campfire(campfire_choice, cards, decks)

            for card_purchased in run["cards_purchased"]:
                if card_purchased["floor"] > floor:
                    break
                if card_purchased["floor"] == floor:
                    self.modify_cards_by_purchase(card_purchased, cards)
            
            decks.append({
                "floor": floor,
                "cards": cards.copy(),
            })
        
        master_deck_sorted = sorted(run["master_deck"])
        generated_master_deck_sorted = sorted(decks[-1]["cards"])

        if self.print_once:
            self.print_once = False
            print(master_deck_sorted)
            print("--------------------------------------------------------------")
            print(generated_master_deck_sorted)

        if master_deck_sorted == generated_master_deck_sorted:
            self.count_master_decks_failed += 1
        else:
            self.count_master_decks += 1

        print(f"Master decks did not match: {self.count_master_decks_failed}")
        print(f"Master decks did match: {self.count_master_decks}")

        return decks

    def modify_cards_by_card_choice(self, card_choice, cards):
        if card_choice["picked"] != "SKIP":
            cards.append(card_choice["picked"])

    def modify_cards_by_event(self, event, cards, decks):
        if "cards_removed" in event:
            for card_removed in event["cards_removed"]:
                if card_removed in cards:
                    cards.remove(card_removed)
                else:
                    for deck in decks:
                        deck["cards"].append(card_removed)
        
        if "cards_transformed" in event:
            for card_transformed in event["cards_transformed"]:
                if card_transformed in cards:
                    cards.remove(card_transformed)
                else:
                    for deck in decks:
                        deck["cards"].append(card_transformed)

        if "cards_obtained" in event:
            for card_obtained in event["cards_obtained"]:
                cards.append(card_obtained)

        if "cards_upgraded" in event:
            for card_upgraded in event["cards_upgraded"]:
                # fulfix: lös "+2"
                # splitted = card_upgraded.split("+")
                # if len(splitted):

                if card_upgraded in cards:
                    cards.remove(card_upgraded)
                else:
                    for deck in decks:
                        deck["cards"].append(card_upgraded)

                cards.append(card_upgraded + "+1")


    def modify_cards_by_campfire(self, campfire_choice, cards, decks):
        if campfire_choice["data"] in cards:
            cards.remove(campfire_choice["data"])
        else:
            for deck in decks:
                deck["cards"].append(campfire_choice["data"])
        cards.append(campfire_choice["data"] + "+1")


    def modify_cards_by_purchase(self, card_purchased, cards):
        cards.append(card_purchased["card"])
