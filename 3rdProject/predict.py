#!/usr/bin/env python
# coding: utf-8

from flask import Flask
from flask import request
from flask import jsonify

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(150, 150))

interpreter = tflite.Interpreter(model_path='garbage-classification-model.tflite')
interpreter.allocate_tensors() # Weights

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
'cardboard',
'glass',
'metal',
'paper',
'plastic',
'trash'
]


#url = 'https://images.kkeu.de/is/image/BEG/Packaging_Supplies/Cardboard_boxes/Folding_cardboard_box_FEFCO_0201_pdplarge-mrd--000026508745_PRD_org_all.jpg'

def run_classification(url):
	X = preprocessor.from_url(url)
	interpreter.set_tensor(input_index, X)
	interpreter.invoke()
	preds = interpreter.get_tensor(output_index)
	
	float_predictions = preds[0].tolist()

	return dict(zip(classes, float_predictions))


app = Flask('predict')  
@app.route('/predict', methods=['POST'])
def garbage_classifier():
    body = request.get_json()  
    url = body['url']
    result = {
      	"garbage_classification": run_classification(url)
    }

    return jsonify(result)

#if __name__ == "__main__":
#  app.run(debug=True, host='0.0.0.0', port=9696, reloader_interval=3)


if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=9696)

