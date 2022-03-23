import copy


class OneHotEncoder:

    def __init__(self, card_identifier):
        self.card_identifier = card_identifier
        self.intial_one_hot_available_choices = None
        self.intial_one_hot_deck = None

    def encode(self, choices):
        card_ids = self.card_identifier.get_card_ids()
        self.create_one_hot_available_choices(card_ids)
        self.create_one_hot_deck(card_ids)

        one_hot_encoded_data = []

        for choice in choices:
            one_hot_available_choices = self.create_one_hot_with_choices(choice["available_choices"])
            one_hot_deck = self.create_one_hot_with_deck(choice["deck"])

            if one_hot_deck is None:
                continue

            one_hot_encoded_data.append({
                "inputs": {
                    "available_choices": one_hot_available_choices,
                    "deck": one_hot_deck,
                },
                "targets": self.create_one_hot_with_choices([choice["player_choice"]])
            })

        return one_hot_encoded_data 

    def create_one_hot_available_choices(self, card_ids):
        self.intial_one_hot_available_choices = []
        for _ in range(len(card_ids)):
            self.intial_one_hot_available_choices.append(0)

    def create_one_hot_deck(self, card_ids):
        self.intial_one_hot_deck = []
        for _ in range(len(card_ids)):
            self.intial_one_hot_deck.append([])
            for _ in range(6):
                self.intial_one_hot_deck[-1].append(0)

    def create_one_hot_with_choices(self, choices):
        one_hot_encoded_choices = self.intial_one_hot_available_choices.copy()
        for choice in choices:
            one_hot_encoded_choices[choice] = 1
        
        return one_hot_encoded_choices

    def create_one_hot_with_deck(self, deck):
        one_hot_encoded_deck = copy.deepcopy(self.intial_one_hot_deck)

        counts = self.intial_one_hot_available_choices.copy()

        for card in deck:
            counts[card] += 1

        for index, count in enumerate(counts):
            if count > 6:
                return None
            if count == 0:
                continue

            one_hot_encoded_deck[index][count - 1] = 1
        
        return one_hot_encoded_deck
