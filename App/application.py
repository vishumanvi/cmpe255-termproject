from flask import Flask, jsonify, render_template, make_response
from flask import request
import pickle
import pandas as pd
import numpy as np
import io
import csv
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from werkzeug import secure_filename
application = app = Flask(__name__)

activities = {1:'lying',2:'sitting', 3:'standing',4:'walking', 5:'running',6:'cycling', 7:'Nordic walking', 9:'watching TV', 10:'computer work 11 – car driving', 12:'ascending stairs', 13:'descending stairs', 16:'vacuum cleaning', 17:'ironing', 18: 'folding laundry', 19:'house cleaning', 20:'playing soccer', 24:'rope jumping'}
@app.route("/")
def home():
	# result = {"name":"Akshay","age":"24"}
	return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['test_file']
		f.save(secure_filename(f.filename))

		return 'file uploaded successfully'
	
@app.route('/test', methods=['POST'])
def test_data(result=None):
	global activities
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
	labels_df = pd.read_csv(labels_file)
	print(test_df.shape)
	print(labels_df.shape)

	predicted_values = clf.predict(test_df)
	counts = np.bincount(predicted_values)
	final_class = np.argmax(counts)
	print("final_class is : {}".format(final_class))

	final_predicted_activity = activities[final_class]
	print("final_predicted_activity is : {}".format(final_predicted_activity))


	try:
		if labels_df is not None:
			accuracy_score_final = accuracy_score(labels_df, predicted_values)
			f1_score_final = f1_score(labels_df, predicted_values, average='macro')
	except UnboundLocalError:
		accuracy_score_final = "Unknown"
		f1_score_final = "Unknown"

	js = {
		"predicted_class":str(final_predicted_activity),
		"accuracy":str(accuracy_score_final),
		"f1score":str(f1_score_final)
		}

	return render_template("index.html",result = js)

if __name__ == "__main__":
	app.run(debug=True)
