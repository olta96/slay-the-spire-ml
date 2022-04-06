import copy


class OneHotEncoder:

    def __init__(self, card_identifier, relic_identifier, max_card_count, should_include_floor, acts):
        self.max_card_count = max_card_count
        self.should_include_floor = should_include_floor
        self.acts = acts
        self.card_identifier = card_identifier
        self.relic_identifier = relic_identifier
        self.intial_one_hot_available_choices = None
        self.intial_one_hot_deck = None
        self.intial_one_hot_relics = None
        self.intial_one_hot_act = None
        self.intial_one_hot_floors = None
        self.max_floor_reached = None

    def encode(self, choices, max_floor_reached):
        card_ids = self.card_identifier.get_card_ids()
        relic_ids = self.relic_identifier.get_relic_ids()
        self.create_one_hot_available_choices(card_ids)
        self.create_one_hot_deck(card_ids)
        self.create_one_hot_relics(relic_ids)
        self.create_one_hot_act()
        if self.should_include_floor:
            self.create_one_hot_floors(max_floor_reached)

        one_hot_encoded_data = []

        for choice in choices:
            one_hot_available_choices = self.create_one_hot_with_choices(choice["available_choices"])
            one_hot_avaiable_relics = self.create_one_hot_with_relics(choice["relics"])
            one_hot_deck = self.create_one_hot_with_deck(choice["deck"])
            one_hot_acts = self.create_one_hot_with_acts(choice["floor"])
            targets = self.create_one_hot_with_choices([choice["player_choice"]])

            if self.should_include_floor:
                floor = self.create_one_hot_with_floor(choice["floor"])

            if one_hot_deck is None:
                continue

            one_hot_encoded = {
                "inputs": {
                    "available_choices": one_hot_available_choices,
                    "relics" : one_hot_avaiable_relics,
                    "deck": one_hot_deck,
                    "acts": one_hot_acts,
                },
                "targets": targets,
            }

            if self.should_include_floor:
                one_hot_encoded["inputs"]["floor"] = floor

            one_hot_encoded_data.append(one_hot_encoded)

        return one_hot_encoded_data 

    def create_one_hot_available_choices(self, card_ids):
        self.intial_one_hot_available_choices = []
        for _ in range(len(card_ids)):
            self.intial_one_hot_available_choices.append(0)

    def create_one_hot_relics(self, relic_ids):
        self.intial_one_hot_relics = []
        for _ in range(len(relic_ids)):
            self.intial_one_hot_relics.append(0)

    def create_one_hot_act(self):
        self.intial_one_hot_act = []
        for _ in range(len(self.acts)):
            self.intial_one_hot_act.append(0)

    def create_one_hot_floors(self, max_floor_reached):
        self.intial_one_hot_floors = []
        for _ in range(max_floor_reached):
            self.intial_one_hot_floors.append(0)

    def create_one_hot_deck(self, card_ids):
        self.intial_one_hot_deck = []
        for _ in range(len(card_ids)):
            self.intial_one_hot_deck.append([])
            for _ in range(self.max_card_count):
                self.intial_one_hot_deck[-1].append(0)

    def create_one_hot_with_choices(self, choices):
        one_hot_encoded_choices = self.intial_one_hot_available_choices.copy()
        for choice in choices:
            one_hot_encoded_choices[choice] = 1
        
        return one_hot_encoded_choices

    def create_one_hot_with_relics(self, relics):
        one_hot_encoded_relics = self.intial_one_hot_relics.copy()
        for relic in relics:
            one_hot_encoded_relics[relic] = 1
        
        return one_hot_encoded_relics

    def create_one_hot_with_acts(self, floor):
        one_hot_encoded_acts = self.intial_one_hot_act.copy()
        for i, act in enumerate(self.acts):
            if floor >= act["from_floor_inclusive"] and floor <= act["to_floor_inclusive"]:
                one_hot_encoded_acts[i] = 1
        
        return one_hot_encoded_acts

    def create_one_hot_with_floor(self, floor):
        one_hot_encoded_floor = self.intial_one_hot_floors.copy()
        one_hot_encoded_floor[floor - 1] = 1
        
        return one_hot_encoded_floor

    def create_one_hot_with_deck(self, deck):
        one_hot_encoded_deck = copy.deepcopy(self.intial_one_hot_deck)

        counts = self.intial_one_hot_available_choices.copy()

        for card in deck:
            counts[card] += 1

        for index, count in enumerate(counts):
            if count > self.max_card_count:
                return None
            if count == 0:
                continue

            one_hot_encoded_deck[index][count - 1] = 1
        
        return one_hot_encoded_deck
