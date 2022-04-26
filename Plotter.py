import json, matplotlib, numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt



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
    
    def __init__(self, name = None):
        if name is not None:
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

        max_loss = 0
        max_accuracy = 0
        y_losses = []
        y_accuracies = []
        x_epochs = []
        labels = []

        for dp in graph_data:
            x_epoch = []
            y_loss = []
            y_accuracy = []

            for row in dp["data"]:
                x_epoch.append(row[0] + 1)
                y_loss.append(row[1])
                y_accuracy.append(row[2])
                max_loss = max(max_loss, row[1])
                max_accuracy = max(max_accuracy, row[2])

            y_losses.append(y_loss)
            y_accuracies.append(y_accuracy)
            x_epochs.append(x_epoch)

            labels.append(dp["name"])

        figure_loss, axes_loss = plt.subplots()

        axes_loss.set_xlabel("Epoch")
        axes_loss.set_ylabel("Loss")
        axes_loss.set_title("Loss")
        axes_loss.set_xticks(np.arange(0, len(x_epochs[0]) + 1, 1))
        for i in range(len(labels)):
            axes_loss.plot(x_epochs[i], y_losses[i], label=f"Case{i+1}")

        axes_loss.legend()

        # save plot
        figure_loss.savefig("loss.png")

        figure_accuracy, axes_accuracy = plt.subplots()

        axes_accuracy.set_xlabel("Epoch")
        axes_accuracy.set_ylabel("Accuracy %")
        axes_accuracy.set_title("Accuracy")
        axes_accuracy.set_xticks(np.arange(0, len(x_epochs[0]) + 1, 1))
        axes_accuracy.set_ylim(0, 100)
        for i in range(len(labels)):
            axes_accuracy.plot(x_epochs[i], y_accuracies[i], label=f"Case{i+1}")

        axes_accuracy.legend()

        # save plot
        figure_accuracy.savefig("accuracy.png")


if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot()
