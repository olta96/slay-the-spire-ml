class RelicIdentifier:
    
    def __init__(self):
        self.relic_ids = []

    def get_relic_ids(self):
        return self.relic_ids

    def identify(self, *relic_names):
        identified_relics = []

        for relic_name in relic_names:
            if relic_name in self.relic_ids:
                identified_relics.append(self.relic_ids.index(relic_name))
            else:
                self.relic_ids.append(relic_name)
                identified_relics.append(len(self.relic_ids) - 1)
        
        if len(relic_names) == 1:
            return identified_relics[0]
        else:
            return identified_relics
