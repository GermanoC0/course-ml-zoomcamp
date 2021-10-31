import pickle
from flask import Flask
from flask import request
from flask import request
from flask import jsonify
import xgboost as xgb

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
	dv, model = pickle.load(f_in)

app = Flask('project_pizza')

@app.route('/predict', methods=['POST'])
def predict():
	pizza = request.get_json()
	
	X = dv.transform([pizza])

	features = dv.get_feature_names()
	DMatrixPred = xgb.DMatrix(X, feature_names=features)

	y_pred = model.predict(DMatrixPred)
	price = y_pred
	
	result = {
		'price': float(price),
	}
	
	return jsonify(result)


if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=9090)

