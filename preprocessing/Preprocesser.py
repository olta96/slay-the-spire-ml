import os
from preprocessing.Logger import Logger
from preprocessing.FileHandler import FileHandler
from preprocessing.RunFilterer import RunFilterer
from preprocessing.CardIdentifier import CardIdentifier
from preprocessing.ChoiceBuilder import ChoiceBuilder
from source_folder_path import source_folder_path
from preprocessing.OneHotEncoder import OneHotEncoder


class Preprocesser:
    
    FILE_CAP = 10

    def __init__(self, one_hot_encoded_json_filename, cards_ids_filename):
        self.one_hot_encoded_json_filename = one_hot_encoded_json_filename
        self.cards_ids_filename = cards_ids_filename
        self.file_handler = FileHandler()
        self.run_filterer = RunFilterer()
        self.card_identifier = CardIdentifier()
        self.choice_builder = ChoiceBuilder(self.card_identifier)
        self.one_hot_encoder = OneHotEncoder(self.card_identifier)

        self.source_filenames = []
        self.loaded_files = []
        self.filtered_runs = []
        self.choices = None
        self.one_hot_encoded_data = None

    def get_one_hot_encoded_data(self):
        return self.one_hot_encoded_data

    def get_card_ids(self):
        return self.card_identifier.get_card_ids()

    def start(self):
        self.build_source_paths()

        Logger.getLogger().log("Started reading json files")
        self.read_source()

        Logger.getLogger().log("Started filtering runs")
        self.filter_loaded_files()
        Logger.getLogger().log(f"Filtered: {len(self.filtered_runs)} runs.")
        Logger.getLogger().log(f"Skipped runs: {self.run_filterer.get_skipped_run_count()}")

        Logger.getLogger().log("Started building choices")
        self.build_choices()
        Logger.getLogger().log(f"Built: {len(self.choices)} choices")

        Logger.getLogger().log("Started one hot encoding choices")
        self.encode_choices()
        Logger.getLogger().log(f"Encoded: {len(self.one_hot_encoded_data)} choices")

        Logger.getLogger().log(f"Writing card IDs to {self.cards_ids_filename}")
        self.write_card_ids()

        Logger.getLogger().log(f"Writing one hot encoding data to {self.one_hot_encoded_json_filename}")
        self.write_one_hot_encoded_data()

    def build_source_paths(self):
        for i, file in enumerate(os.listdir(source_folder_path)):
            if i % 10 == 0:
                Logger.getLogger().log("Loaded", i, "files")
            self.source_filenames.append(file)
            if i == self.FILE_CAP:
                break

    def read_source(self):
        for i, source_filename in enumerate(self.source_filenames):
            if i % 10 == 0:
                Logger.getLogger().log(i, "files read")
            source_file_path = source_folder_path + "/" + source_filename
            self.loaded_files.append(self.file_handler.read_json(source_file_path))

    def filter_loaded_files(self):
        for runs in self.loaded_files:
            for run_dict in runs:
                unfiltered_run = run_dict["event"]
                filtered_run = self.run_filterer.get_filtered_run(unfiltered_run)
                if filtered_run is not None:
                    self.filtered_runs.append(filtered_run)

    def build_choices(self):
        self.choices = self.choice_builder.build(self.filtered_runs)

    def encode_choices(self):
        self.one_hot_encoded_data = self.one_hot_encoder.encode(self.choices)

    def write_card_ids(self):
        self.file_handler.write_json(self.cards_ids_filename, self.get_card_ids())

    def write_one_hot_encoded_data(self):
        self.file_handler.write_json(self.one_hot_encoded_json_filename, self.one_hot_encoded_data)
