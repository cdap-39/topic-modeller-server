from flask import Flask
from flask import request
from flask import jsonify
import os

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import lda


@app.route("/ping")
def ping():
    return "pong"


@app.route('/process', methods=['POST'])
def json_example():
    req_data = request.get_json()
    os.system(lda)
    return jsonify(req_data)


app.run(host="0.0.0.0", port="8082")