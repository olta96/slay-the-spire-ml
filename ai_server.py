import flask, json



from preprocessing.CardIdentifier import CardIdentifier
from preprocessing.RelicIdentifier import RelicIdentifier
from preprocessing.OneHotEncoder import OneHotEncoder

def setup_one_hot_encoder():
    def read_config_json():
        with open("config.json") as config_file:
            return json.loads(config_file.read())

    config_options = read_config_json()

    def read_card_ids():
        with open(config_options["card_ids_json_filename"]) as card_ids_json_file:
            return json.loads(card_ids_json_file.read())

    def read_relic_ids():
        with open(config_options["relic_ids_json_filename"]) as relic_ids_json_file:
            return json.loads(relic_ids_json_file.read())

    card_ids = read_card_ids()
    relic_ids = read_relic_ids()

    card_identifier = CardIdentifier(card_ids)
    relic_identifier = RelicIdentifier(relic_ids)
    one_hot_encoder = OneHotEncoder(card_identifier, relic_identifier, config_options["deck_max_card_count"])

    return card_identifier, relic_identifier, one_hot_encoder





card_identifier, relic_identifier, one_hot_encoder = setup_one_hot_encoder()

app = flask.Flask(__name__)

@app.route('/make_choice', methods=["POST"])
def make_choice():
    data = flask.request.get_json()
    print(data)
    return flask.jsonify({"message": "success"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
