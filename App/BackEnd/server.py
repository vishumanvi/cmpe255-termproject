from flask import Flask, jsonify
from flask import request
import pickle
import pandas as pd
import numpy as np
import io
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return 'OK'


@app.route('/test', methods=['POST'])
def test_data():
    model_file_path = "../models/"
    modelname = request.args.get('modelname')
    dim_red = request.args.get('dim_red')

    fil = request.files['test_file']
    stream = io.StringIO(fil.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    lines = list(csv_input)

    with open('../testdata/test_file.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)

    writeFile.close()

    print("Received {} and {} from request".format(modelname, dim_red))
    model_file = "{}{}_{}.pkl".format(model_file_path, modelname, dim_red)
    print("Model file is {}".format(model_file))

    clf = pickle.load(open(model_file, "rb"))

    test_df = pd.read_csv("../testdata/test_file.csv")
    print(test_df.shape)

    predicted_values = clf.predict(test_df)

    counts = np.bincount(predicted_values)
    final_class = np.argmax(counts)

    return jsonify(predicted_class=str(final_class))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)
