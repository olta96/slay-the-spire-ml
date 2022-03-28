import os
from preprocessing.Logger import Logger
from preprocessing.FileHandler import FileHandler
from preprocessing.RunFilterer import RunFilterer
from preprocessing.CardIdentifier import CardIdentifier
from preprocessing.ChoiceBuilder import ChoiceBuilder
from source_folder_path import source_folder_path
from preprocessing.OneHotEncoder import OneHotEncoder

class Preprocesser:
    
    FILE_CAP = 200
    LOGS_FILENAME = "preprocessor_logs.txt"
    DECK_MAX_CARD_COUNT = 6
    FILE_BATCH_SIZE = 10

    def __init__(self, one_hot_encoded_json_filename, cards_ids_filename):
        self.one_hot_encoded_json_filename = one_hot_encoded_json_filename
        self.cards_ids_filename = cards_ids_filename
        self.file_handler = FileHandler()
        self.run_filterer = RunFilterer()
        self.card_identifier = CardIdentifier()
        self.choice_builder = ChoiceBuilder(self.card_identifier)
        self.one_hot_encoder = OneHotEncoder(self.card_identifier, self.DECK_MAX_CARD_COUNT)

        self.source_filenames = []
        self.filtered_runs = []
        self.choices = None
        self.one_hot_encoded_data = None

    def get_one_hot_encoded_data(self):
        return self.one_hot_encoded_data

    def get_card_ids(self):
        return self.card_identifier.get_card_ids()

    def get_deck_max_card_count(self):
        return self.DECK_MAX_CARD_COUNT

    def start(self):
        self.build_source_paths()

        Logger.get_logger().log("Started reading and filtering json files")
        self.read_and_filter_source()

        Logger.get_logger().log(f"Filtered: {len(self.filtered_runs)} runs.")
        Logger.get_logger().log(f"Skipped runs: {self.run_filterer.get_skipped_run_count()}")

        Logger.get_logger().log("Started building choices")
        self.build_choices()
        Logger.get_logger().log(f"Built: {len(self.choices)} choices")

        Logger.get_logger().log("Started one hot encoding choices")
        self.encode_choices()
        Logger.get_logger().log(f"Encoded: {len(self.one_hot_encoded_data)} choices")

        Logger.get_logger().log(f"Writing card IDs to {self.cards_ids_filename}")
        self.write_card_ids()

        Logger.get_logger().log(f"Writing one hot encoding data to {self.one_hot_encoded_json_filename}")
        self.write_one_hot_encoded_data()

        Logger.get_logger().log(f"Writing logs to {self.LOGS_FILENAME}")
        self.write_logs()

    def build_source_paths(self):
        for i, file in enumerate(os.listdir(source_folder_path)):
            self.source_filenames.append(file)
            if i == self.FILE_CAP:
                break

    def read_and_filter_source(self):
        loaded_files = []
        for i, source_filename in enumerate(self.source_filenames):
            if i % self.FILE_BATCH_SIZE == 0:
                Logger.get_logger().log(i, "files read")
                self.filter_loaded_files(loaded_files)
                loaded_files = []
            source_file_path = source_folder_path + "/" + source_filename
            loaded_files.append(self.file_handler.read_json(source_file_path))
        self.filter_loaded_files(loaded_files)

    def filter_loaded_files(self, loaded_files):
        for runs in loaded_files:
            for run_dict in runs:
                unfiltered_run = run_dict["event"]
                filtered_run = None
                try:
                    filtered_run = self.run_filterer.get_filtered_run(unfiltered_run)
                except Exception as e:
                    Logger.get_logger().log("Failed to filter a run. (skipping)", e)
                if filtered_run is not None:
                    self.filtered_runs.append(filtered_run)
        Logger.get_logger().log(f"{len(loaded_files)} files filtered")

    def build_choices(self):
        self.choices = self.choice_builder.build(self.filtered_runs)

    def encode_choices(self):
        self.one_hot_encoded_data = self.one_hot_encoder.encode(self.choices)

    def write_card_ids(self):
        self.file_handler.write_json(self.cards_ids_filename, self.get_card_ids(), indent=4)

    def write_one_hot_encoded_data(self):
        self.file_handler.write_json(self.one_hot_encoded_json_filename, self.one_hot_encoded_data)

    def write_logs(self):
        self.file_handler.write(self.LOGS_FILENAME, Logger.get_logger().get_log_messages())
