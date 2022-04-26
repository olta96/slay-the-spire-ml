import json



class Plotter:
    """
    - 2 plots
    - line graph
    - Malformed runs and strict runs.
    - Accuracy
    - Loss
    - Same color for same models.
    - x-axis: epochs
    - y-axis: loss & accuracy
    - barchart - amount of legendary cards chosen
    """
    
    def __init__(self, name):
        self.name = name
        self.to_save = []

    def push_epoch(self, epoch, loss, accuracy):
        self.to_save.append((epoch, loss, accuracy))

    def save(self):
        graph_data = self.read_graph_data()
        
        graph_data.append({
            "name": self.name,
            "data": self.to_save,
        })

        self.write_graph_data(graph_data)

    def read_graph_data(self):
        with open("graph_data.json", "r") as json_file:
            graph_data = json.loads(json_file.read())
        return graph_data

    def write_graph_data(self, graph_data):
        with open("graph_data.json", "w") as json_file:
            json_file.write(json.dumps(graph_data, indent=4))

    def plot(self):
        graph_data = self.read_graph_data()
