import enum
from preprocessing.reference_list import curse_cards

MAX_FLOOR = 80

class MasterDeckMatchingOption(enum.Enum):
    EXACT_MATCH = enum.auto()
    SIMILAR = enum.auto()

class DeckBuilder:

    DECK_MATCHING_OPTION = MasterDeckMatchingOption.EXACT_MATCH

    def __init__(self):
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
        ]

    def build(self, run):
        decks = []
        cards = self.starting_deck.copy()
        for floor in range(MAX_FLOOR):
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

            for card_purged in run["cards_purged"]:
                if card_purged["floor"] > floor:
                    break
                if card_purged["floor"] == floor:
                    self.modify_cards_by_purge(card_purged, cards, decks)
            
            decks.append({
                "floor": floor,
                "cards": cards.copy(),
            })
        
        original_master_deck = run["master_deck"]
        generated_master_deck = decks[-1]["cards"]

        if self.decks_match(original_master_deck, generated_master_deck):
            return decks
        elif self.DECK_MATCHING_OPTION == MasterDeckMatchingOption.SIMILAR and self.decks_have_similar_length(original_master_deck, generated_master_deck):
            return decks
        else:
            return None

    def modify_cards_by_card_choice(self, card_choice, cards):
        if card_choice["picked"] != "SKIP":
            cards.append(card_choice["picked"])

    def modify_cards_by_event(self, event, cards, decks):
        if "cards_removed" in event:
            for card_removed in event["cards_removed"]:
                self.remove_card(card_removed, cards, decks)
        
        if "cards_transformed" in event:
            for card_transformed in event["cards_transformed"]:
                self.remove_card(card_transformed, cards, decks)

        if "cards_obtained" in event:
            for card_obtained in event["cards_obtained"]:
                if card_obtained not in curse_cards:
                    cards.append(card_obtained)

        if "cards_upgraded" in event:
            for card_upgraded in event["cards_upgraded"]:
                self.remove_card(card_upgraded, cards, decks)
                cards.append(card_upgraded + "+1")


    def modify_cards_by_campfire(self, campfire_choice, cards, decks):
        self.remove_card(campfire_choice["data"], cards, decks)
        if "+" in campfire_choice["data"]:
            splitted = campfire_choice["data"].split("+")
            cards.append(splitted[0] + "+" + str(int(splitted[1]) + 1))
        else:
            cards.append(campfire_choice["data"] + "+1")

    def modify_cards_by_purchase(self, card_purchased, cards):
        cards.append(card_purchased["card"])

    def modify_cards_by_purge(self, card_purged, cards, decks):
        self.remove_card(card_purged["card"], cards, decks)

    def remove_card(self, to_remove, cards, decks):
        if to_remove in cards:
            cards.remove(to_remove)
        elif to_remove not in curse_cards:
            for deck in decks:
                deck["cards"].append(to_remove)

    def decks_have_similar_length(self, deck_a, deck_b):
        deck_a_copy = deck_a.copy()
        deck_b_copy = deck_b.copy()

        for val in deck_a_copy:
            if val in curse_cards:
                deck_a_copy.remove(val)

        for val in deck_b_copy:
            if val in curse_cards:
                deck_b_copy.remove(val)

        return abs(len(deck_a_copy) - len(deck_b_copy)) < 2

    def decks_match(self, deck_a, deck_b):
        sorted_a = sorted(deck_a)
        sorted_b = sorted(deck_b)

        for val in sorted_a:
            if val in curse_cards:
                sorted_a.remove(val)

        for val in sorted_b:
            if val in curse_cards:
                sorted_b.remove(val)

        if len(sorted_a) != len(sorted_b):
            return False

        for i in range(len(sorted_a)):
            if sorted_a[i] != sorted_b[i]:
                return False
        
        return True
