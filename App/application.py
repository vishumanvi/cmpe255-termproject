from flask import Flask, jsonify
from flask import request
import pickle
import pandas as pd
import numpy as np
import io
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

application = app = Flask(__name__)

@app.route('/')
def index():
    return 'OK'

@app.route('/test', methods=['POST'])
def test_data():
    modelname = request.args.get('modelname')
    dim_red = request.args.get('dim_red')
    print("Received {} and {} from request".format(modelname, dim_red))

    fil = request.files['test_file']
    stream = io.StringIO(fil.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    print("stream is {}".format(stream))

    print("csv_input is {}".format(csv_input))
    lines = list(csv_input)

    with open('test_file.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)

    writeFile.close()
    model_file = "{}_{}.pkl".format(modelname, dim_red)
    print("Model file is {}".format(model_file))
    clf = pickle.load(open(model_file, "rb"))
    test_df = pd.read_csv("test_file.csv")
    labels_file = fil.filename
    labels_file = labels_file.replace("activity", "label")
    print("New labels_file : {} ".format(labels_file))
    labels_file = labels_file
    print(labels_file)
    try:
        labels_df = pd.read_csv(labels_file)

    except FileNotFoundError as e:
        print(e)

    print(test_df.shape)
    predicted_values = clf.predict(test_df)
    counts = np.bincount(predicted_values)
    final_class = np.argmax(counts)

    try:
        if labels_df is not None:
            accuracy_score_final = accuracy_score(labels_df, predicted_values)
            f1_score_final = f1_score(labels_df, predicted_values, average='macro')
    except UnboundLocalError:
        accuracy_score_final = "Unknown"
        f1_score_final = "Unknown"

    return jsonify(predicted_class=str(final_class), accuracy=str(accuracy_score_final), f1score=str(f1_score_final))

if __name__ == "__main__":
    app.run(debug=True)
