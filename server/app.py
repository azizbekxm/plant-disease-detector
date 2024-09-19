import os
from flask import Flask, render_template, request
import pandas as pd

import torch
from torchvision import models

import predict

# Get the directory path of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the file
disease_info_path = os.path.join(script_directory, 'disease_info.csv')
supplement_info_path = os.path.join(script_directory, 'supplement_info.csv')

disease_info = pd.read_csv(disease_info_path, encoding='utf-8')
supplement_info = pd.read_csv(supplement_info_path, encoding='utf-8')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    load_model = models.vgg19(pretrained=True)
    load_model.classifier = checkpoint['classifier']
    load_model.load_state_dict(checkpoint['state_dict'])
    load_model.class_to_idx = checkpoint['class_to_idx']
    load_model.idx_to_class = checkpoint['idx_to_class']
    return load_model


model = load_checkpoint(os.path.join(script_directory, 'cotton-disease-detection-best.pth'))


app = Flask(__name__)
# Set the path for file uploads
UPLOAD_FOLDER = os.path.join(script_directory, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_url = os.path.join('static', 'uploads', filename)
        # Create the 'static/uploads' directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        image.save(file_path)
        print(file_path)

        # possibility and class name
        ps, class_name = predict.predict(model, 'gpu', 1, file_path)

        predict_idx = int(class_name)
        ps = round(ps * 100, 2)
        title = disease_info['disease_name'][predict_idx]
        description = disease_info['description'][predict_idx]
        prevent = disease_info['Possible Steps'][predict_idx]

        supplement_name = supplement_info['supplement name'][predict_idx]
        supplement_image_url = supplement_info['supplement image'][predict_idx]
        supplement_buy_link = supplement_info['buy link'][predict_idx]
        return render_template('submit.html', title=title, confidence=ps, desc=description, prevent=prevent,
                               image_url=image_url, pred=predict_idx, sname=supplement_name, simage=supplement_image_url,
                               buy_link=supplement_buy_link)


if __name__ == '__main__':
    app.run(debug=True)
