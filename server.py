from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import model


app = Flask(__name__, static_url_path="/static")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # 画像として読み込み
    stream = request.files['image'].stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    num_data = model.predict_digit(img)

    return render_template('result.html')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8888)
