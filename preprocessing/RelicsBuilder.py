class RelicsBuilder:
    
    BOSS_RELIC_CHOICES = [
        {
            "floor": 0,
            "index": 0,
        },
        {
            "floor": 0,
            "index": 1,
        },
        {
            "floor": 0,
            "index": 2,
        },
    ]

    def __init__(self, relics_exact_match):
        self.relics_exact_match = relics_exact_match
        self.starting_relics = [
            "Burning Blood",
        ]
    
    def build(self, run):
        relics_by_floor = []
        relics = self.starting_relics.copy()

        for floor in range(run["floor_reached"] + 1):
            for relic in run["relics_obtained"]:
                if relic["floor"] > floor:
                    break
                if relic["floor"] == floor:
                    relics.append(relic["key"])
            
            for event_choice in run["event_choices"]:
                if event_choice["floor"] > floor:
                    break
                if event_choice["floor"] == floor:
                    self.modify_relics_by_event(event_choice, relics, relics_by_floor)

            for relic_purchased in run["relics_purchased"]:
                if relic_purchased["floor"] > floor:
                    break
                if relic_purchased["floor"] == floor:
                    self.modify_relics_by_purchase(relic_purchased, relics)

            for boss_relic_choice in self.BOSS_RELIC_CHOICES:
                if boss_relic_choice["floor"] == floor and len(run["boss_relics"]) > boss_relic_choice["index"]:
                    if "picked" in run["boss_relics"][boss_relic_choice["index"]]:
                        relics.append(run["boss_relics"][boss_relic_choice["index"]]["picked"])

            relics_by_floor.append({
                "floor": floor,
                "relics": relics.copy(),
            })

        original_relics = run["relics"]
        generated_relics = relics_by_floor[-1]["relics"]

        if self.relics_match(original_relics, generated_relics) or\
        not self.relics_exact_match and self.relics_have_similar_length(original_relics, generated_relics):
            return relics_by_floor
        else:
            return None

    def modify_relics_by_event(self, event, relics, relics_by_floor):
        if "relics_lost" in event:
            for relic in event["relics_lost"]:
                self.remove_relic(relic, relics, relics_by_floor)
        
        if "relics_obtained" in event:
            for relic in event["relics_obtained"]:
                relics.append(relic)

    def modify_relics_by_purchase(self, relic_purchased, relics):
        relics.append(relic_purchased["relic"])

    def remove_relic(self, to_remove, relics, relics_by_floor):
        if to_remove in relics:
            relics.remove(to_remove)
        else:
            for relic_floor in relics_by_floor:
                relic_floor["relics"].append(to_remove)

    def relics_have_similar_length(self, relics_a, relics_b):
        return abs(len(relics_a) - len(relics_b)) <= 1

    def relics_match(self, relics_a, relics_b):
        sorted_a = sorted(relics_a)
        sorted_b = sorted(relics_b)

        if len(sorted_a) != len(sorted_b):
            return False
        
        for i in range(len(sorted_a)):
            if sorted_a[i] != sorted_b[i]:
                return False
        
        return True
