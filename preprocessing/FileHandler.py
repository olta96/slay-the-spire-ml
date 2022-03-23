import json

class FileHandler:

    def read_json(self, file_path):
        with open(file_path, "r") as json_file:
            return json.loads(json_file.read())

    def write_json(self, filename, to_write, indent=0):
        with open(filename, "w+") as json_file:
            json_file.write(json.dumps(to_write, indent=indent))

    def write(self, filename, to_write):
        with open(filename, "w+") as write_file:
            write_file.write(to_write)
