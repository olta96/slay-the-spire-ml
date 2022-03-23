from preprocessing.reference_list import AVAILABLE_CARDS

CHARACTER = "IRONCLAD"
MIN_ASCENSION_LEVEL = 10
MIN_FLOOR_REACHED = 44
ENDLESS = False
ASCENSION_MODE = True
CHOSE_SEED = False

def matches_character_chosen(character_chosen):
    return character_chosen == CHARACTER

def matches_ascension_level(ascension_level):
    return ascension_level >= MIN_ASCENSION_LEVEL

def matches_floor_reached(floor_reached):
    return floor_reached >= MIN_FLOOR_REACHED 

def matches_is_endless(is_endless):
    return is_endless == ENDLESS

def matches_is_ascension_mode(is_ascension_mode):
    return is_ascension_mode == ASCENSION_MODE

def matches_chose_seed(chose_seed):
    return chose_seed == CHOSE_SEED



class RunFilterer:

    def __init__(self) -> None:
        self.skipped_run_count = 0
        self.filters = {
            "character_chosen": matches_character_chosen,
            "ascension_level": matches_ascension_level,
            "floor_reached": matches_floor_reached,
            "is_endless": matches_is_endless,
            "is_ascension_mode": matches_is_ascension_mode,
            "chose_seed": matches_chose_seed
        }

    def get_filtered_run(self, unfiltered_run):
        if self.run_matches_filters(unfiltered_run) and not self.is_malformed(unfiltered_run):
            filtered_run = self.create_run(unfiltered_run)
            return filtered_run
        else:
            return None

    def run_matches_filters(self, run):
        for key in self.filters:
            if not self.filters[key](run[key]):
                return False
        
        return True

    def is_malformed(self, unfiltered_run):
        """
        Counts if unfiltered run was malformed:
            len(unfiltered_run["items_purchased"]) != len(unfiltered_run["item_purchase_floors"])
        """
        if len(unfiltered_run["items_purchased"]) != len(unfiltered_run["item_purchase_floors"]):
            self.skipped_run_count += 1
            return True

        return False

    def create_run(self, unfiltered_run):
        return {
            "play_id": unfiltered_run["play_id"],
            "victory": unfiltered_run["victory"],
            "ascension_level": unfiltered_run["ascension_level"],
            "floor_reached": unfiltered_run["floor_reached"],
            "card_choices": unfiltered_run["card_choices"],
            "master_deck": unfiltered_run["master_deck"],
            "cards_purchased": self.get_cards_purchased(unfiltered_run),
            "cards_purged": self.get_cards_purged(unfiltered_run),
            "event_choices": self.get_event_choices(unfiltered_run),
            "campfire_choices": self.get_campfire_choices(unfiltered_run),
        }

    def get_cards_purchased(self, unfiltered_run):
        cards_purchased = []
        for i in range(len(unfiltered_run["items_purchased"])):
            if unfiltered_run["items_purchased"][i] in AVAILABLE_CARDS:
                cards_purchased.append({
                    "card": unfiltered_run["items_purchased"][i],
                    "floor": unfiltered_run["item_purchase_floors"][i],
                })
        return cards_purchased

    def get_cards_purged(self, unfiltered_run):
        cards_purged = []
        for i in range(len(unfiltered_run["items_purged"])):
            if unfiltered_run["items_purged"][i] in AVAILABLE_CARDS:
                cards_purged.append({
                    "card": unfiltered_run["items_purged"][i],
                    "floor": unfiltered_run["items_purged_floors"][i],
                })
        return cards_purged

    def get_event_choices(self, unfiltered_run):
        event_choices = []
        for event_choice in unfiltered_run["event_choices"]:
            self.append_event_choice(event_choices, event_choice)
        return event_choices
        
    def get_campfire_choices(self, unfiltered_run):
        campfire_choices = []
        for campfire_choice in unfiltered_run["campfire_choices"]:
            if campfire_choice["key"] == "SMITH":
                campfire_choices.append(campfire_choice)
        return campfire_choices
        
    def append_event_choice(self, event_choices, event_choice):
        if any(key in event_choice.keys() for key in ["cards_obtained", "cards_removed", "cards_upgraded"]):
            new_event_choice = {}
            for key in ["cards_obtained", "cards_transformed", "cards_removed", "cards_upgraded", "floor"]:
                if key in event_choice.keys():
                    new_event_choice[key] = event_choice[key]
            event_choices.append(new_event_choice)

    def get_skipped_run_count(self):
        return self.skipped_run_count
