from flask import Flask, jsonify, request
from torch_utils import transform_image, get_prediction, data_classes

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/check", methods=['POST'])
def check_class():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'no file'})
        elif not allowed_file(file.filename):
            return jsonify({'error': 'not allowed extension'})

        try:
            image_bytes = file.read()
            tensor_img = transform_image(image_bytes)
            prediction = get_prediction(tensor_img)
            data = {'prediction:': prediction.item(), 'class_name': str(data_classes[prediction.item()])}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

    return jsonify({'error': '00'})

