class RelicIdentifier:
    
    def __init__(self, relic_ids=None):
        if relic_ids is None:
            self.relic_ids = []
        else:
            self.relic_ids = relic_ids

    def get_relic_ids(self):
        return self.relic_ids

    def identify(self, *relic_names, always_return_list=False):
        identified_relics = []

        for relic_name in relic_names:
            if relic_name in self.relic_ids:
                identified_relics.append(self.relic_ids.index(relic_name))
            else:
                self.relic_ids.append(relic_name)
                identified_relics.append(len(self.relic_ids) - 1)
        
        if always_return_list:
            return identified_relics

        if len(relic_names) == 1:
            return identified_relics[0]
        else:
            return identified_relics
