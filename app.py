import json

import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
import urllib.request
import os
from PIL import Image
from numpy.linalg import norm
import torch
from werkzeug.utils import secure_filename
import clip
from sklearn.neighbors import NearestNeighbors

feature_list = pickle.load(open("static/uploads/features-lostpet-clip.pickle", "rb"))
filenames = ["dogs good hands/" + x for x in os.listdir("static/uploads/dogs good hands/")]

with open("dog_data.json", "r") as f:
    dogs_data = json.load(f)
# print(filenames)

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',
                             metric='euclidean').fit(feature_list)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def extract_features_from_image(img_path, model, preprocess):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features


def extract_features_from_text(text, model, preprocess):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features


def find_similar(model_input, model, preprocess, input_type="photo"):
    if input_type == "photo":
        features = extract_features_from_image(model_input, model, preprocess)
    else:
        features = extract_features_from_text(model_input, model, preprocess)
    features = torch.cat((features, features), axis=-1)
    print(features.shape)
    distances, indices = neighbors.kneighbors([features[0].tolist()])
    return display_image([filenames[x] for x in indices[0, :3]])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return find_similar(path, model, preprocess)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/text', methods=['POST'])
def upload_text():
    text = request.form['text']
    if text == '':
        flash('No text selected for uploading')
        return redirect(request.url)
    else:
        # print('upload_image filename: ' + filename)
        flash('Text successfully uploaded and finding....')
        return find_similar(text, model, preprocess, "text")


def prettify_dict(dct):
    s = ""
    for key, value in dct.items():
        if key == "img_href":
            continue
        s += f"{key}: {value}. "
    return s.strip()


@app.route('/display/<filename>')
def display_image(filenames):
    # print('display_image filename: ' + filename)

    filenames , dog_index = ["static/uploads/" + x for x in filenames], [int(x.split("_")[-1].strip(".jpg")) for x in filenames]
    print(filenames)
    return render_template("showing.html", dogs_image=filenames, descriptions=[f"Собака {i + 1}. {prettify_dict(dogs_data[index])}" for i, index in enumerate(dog_index)])


if __name__ == "__main__":
    app.run()



# import json
#
# import numpy as np
# from flask import Flask, flash, request, redirect, url_for, render_template
# import pickle
# import urllib.request
# import os
# from PIL import Image
# from numpy.linalg import norm
# import torch
# from werkzeug.utils import secure_filename
# import clip
# from sklearn.neighbors import NearestNeighbors
#
# feature_list = pickle.load(open("static/uploads/features-lostpet-clip.pickle", "rb"))
# filenames = ["dogs good hands/" + x for x in os.listdir("static/uploads/dogs good hands/")]
#
# with open("dog_data.json", "r") as f:
#     dogs_data = json.load(f)
# print(filenames)
#
# neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',
#                              metric='euclidean').fit(feature_list)
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = 'static/uploads/'
#
# app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
#
# def extract_features_from_image(img_path, model, preprocess):
#     image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#     with torch.no_grad():
#         features = model.encode_image(image)
#     return features
#
#
# def extract_features_from_text(text, model, preprocess):
#     text = clip.tokenize([text]).to(device)
#     with torch.no_grad():
#         text_features = model.encode_text(text)
#     return text_features
#
#
# def find_similar(model_input, model, preprocess, input_type="photo"):
#     if input_type == "photo":
#         features = extract_features_from_image(model_input, model, preprocess)
#     else:
#         features = extract_features_from_text(model_input, model, preprocess)
#     features = torch.cat((features, features), axis=-1)
#     print(features.shape)
#     distances, indices = neighbors.kneighbors([features[0].tolist()])
#     return display_image(filenames[indices[0, 0]])
#
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/image', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No image selected for uploading')
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(path)
#         # print('upload_image filename: ' + filename)
#         flash('Image successfully uploaded and displayed below')
#         return find_similar(path, model, preprocess)
#     else:
#         flash('Allowed image types are - png, jpg, jpeg, gif')
#         return redirect(request.url)
#
#
# @app.route('/text', methods=['POST'])
# def upload_text():
#     text = request.form['text']
#     if text == '':
#         flash('No text selected for uploading')
#         return redirect(request.url)
#     else:
#         # print('upload_image filename: ' + filename)
#         flash('Text successfully uploaded and finding....')
#         return find_similar(text, model, preprocess, "text")
#
#
# def prettify_dict(dct):
#     s = ""
#     for key, value in dct.items():
#         if key == "img_href":
#             continue
#         s += f"{key}: {value}. "
#     return s.strip()
#
#
# @app.route('/display/<filename>')
# def display_image(filename):
#     # print('display_image filename: ' + filename)
#     filename, dog_index = "static/uploads/" + filename, int(filename.split("_")[-1].strip(".jpg"))
#     print(filename)
#     return render_template("showing.html", dog_image=filename, description=prettify_dict(dogs_data[dog_index]))
#
#
# if __name__ == "__main__":
#     app.run()
