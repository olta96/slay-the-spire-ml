class CardIdentifier:
    
    def __init__(self):
        self.card_ids = ["SKIP"]

    def get_card_ids(self):
        return self.card_ids

    def identify(self, *card_names):
        identified_cards = []

        for card_name in card_names:
            if card_name in self.card_ids:
                identified_cards.append(self.card_ids.index(card_name))
            else:
                self.card_ids.append(card_name)
                identified_cards.append(len(self.card_ids) - 1)
        
        if len(card_names) == 1:
            return identified_cards[0]
        else:
            return identified_cards
