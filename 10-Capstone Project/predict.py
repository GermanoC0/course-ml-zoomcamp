import pickle
from flask import Flask
from flask import request
from flask import request
from flask import jsonify
import xgboost as xgb

model_file = 'model.bin'

# Load model and DV from bin file
with open(model_file, 'rb') as f_in:
	dv, model = pickle.load(f_in)

app = Flask('real_or_fake_job')

# Service
@app.route('/classify', methods=['POST'])
def predict():
	job_post = request.get_json()
	
	X = dv.transform([job_post])

	features = dv.get_feature_names()
	DMatrixPred = xgb.DMatrix(X, feature_names=features)
	
	# Classify the job_post data and return the probability of fraudulent job
	y_pred = model.predict(DMatrixPred)
	fake_prob = y_pred
	
	result = {
		'fraudulent': float(fake_prob),
	}
	
	return jsonify(result)



if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=9090)
