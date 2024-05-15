from flask import Flask, jsonify, request
from flask_cors import CORS
from util import getResutl
import numpy as np
import json

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Hello, world!"


@app.route("/api/data", methods=["GET", "POST"])
def handle_data():
    if request.method == "POST":
        data = request.get_json()  # Parse JSON data incoming from the request body

        print(np.array(data["c"]))
        # data["result"] = data["c"]
        # data["total_cost"] = 123
        m = int(data["n"])
        n = int(data["m"])
        c = np.array(data["c"])
        a = np.array(data["a"])
        b = np.array(data["b"])
        X, total_cost = getResutl(m, n, c, a, b)
        array = np.array(X)
        array_list = array.tolist()
        data["result"] = array_list
        data["total_cost"] = total_cost

        return jsonify(data), 201
    else:
        data = {"message": "Send me some data!"}
        return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
