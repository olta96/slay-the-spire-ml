import flask

app = flask.Flask(__name__)

@app.route('/make_choice', methods=["POST"])
def make_choice():
    data = flask.request.get_json()
    return flask.jsonify({"message": "success"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
