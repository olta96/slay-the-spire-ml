from preprocessing.RelicsBuilder import RelicsBuilder
from preprocessing.reference_list import AVAILABLE_CARDS
from preprocessing.DeckBuilder import DeckBuilder

AVAILABLE_CHOICES = AVAILABLE_CARDS + ["SKIP"]

class ChoiceBuilder:
    
    def __init__(self, card_identifier, relic_identifier, config_options):
        self.card_identifier = card_identifier
        self.relic_identifier = relic_identifier
        self.deck_builder = DeckBuilder(config_options["strict_mode_decks"])
        self.relics_builder = RelicsBuilder(config_options["strict_mode_relics"])

    def build(self, filtered_runs):
        choices = []

        for filtered_run in filtered_runs:

            decks = self.deck_builder.build(filtered_run)
            if decks is None:
                continue

            relics = self.relics_builder.build(filtered_run)
            if relics is None:
                continue

            for card_choice in filtered_run["card_choices"]:
                if self.is_malformed(card_choice):
                    continue
                
                new_choice = self.build_card_choice(card_choice, decks, relics)
                if new_choice is None:
                    continue

                choices.append(new_choice)
        
        return choices

    def build_card_choice(self, card_choice, decks, relics):
        choices = []

        if self.user_chose_skip(card_choice):
            choices.append(self.card_identifier.identify(card_choice["not_picked"][0]))
            choices.append(self.card_identifier.identify(card_choice["not_picked"][1]))
            choices.append(self.card_identifier.identify(card_choice["not_picked"][2]))
        else:
            choices.append(self.card_identifier.identify(card_choice["not_picked"][0]))
            choices.append(self.card_identifier.identify(card_choice["not_picked"][1]))
            choices.append(self.card_identifier.identify(card_choice["picked"]))

        choices.append(self.card_identifier.identify("SKIP"))

        cards_by_floor = self.get_deck_by_floor(decks, card_choice["floor"])
        if cards_by_floor is None:
            return None

        relics_by_floor = self.get_relics_by_floor(relics, card_choice["floor"])
        if relics_by_floor is None:
            return None

        return {
            "available_choices": choices,
            "player_choice": self.card_identifier.identify(card_choice["picked"]),
            "deck": self.card_identifier.identify(*cards_by_floor),
            "relics": self.relic_identifier.identify(*relics_by_floor)
        }

    def is_malformed(self, card_choice):
        for not_picked in card_choice["not_picked"]:
            if not_picked not in AVAILABLE_CHOICES:
                return True

        return (len(card_choice["not_picked"]) < 2 or len(card_choice["not_picked"]) > 3)\
            or (len(card_choice["not_picked"]) == 3 and card_choice["picked"] != "SKIP")\
            or card_choice["picked"] not in AVAILABLE_CHOICES\
            or "SKIP" in card_choice["not_picked"]

    def get_deck_by_floor(self, decks, floor):
        for deck in decks:
            if deck["floor"] == floor:
                return deck["cards"]
        
        return None

    def get_relics_by_floor(self, relics, floor):
        for relics_and_floor in relics:
            if relics_and_floor["floor"] == floor:
                return relics_and_floor["relics"]
        
        return None

    def user_chose_skip(self, card_choice):
        return len(card_choice["not_picked"]) == 3
