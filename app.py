import os
from flask import Flask, render_template, request, jsonify, redirect
import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch

# Model import
def get_model():
    model = models.densenet121(pretrained = True)
    fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 5),
        nn.LogSoftmax(dim = 1)
    )
    model.classifier = fc
    model.load_state_dict(torch.load('Smap_predictor.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

# Transform image
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# Obtain class list
model_classes = ['Goro_Inagaki', 'Masahiro_Nakai', 'Shingo_Katori', 'Takuya_Kimura', 'Tsuyoshi_Kusanagi']
model_classes = {idx: each for idx, each in enumerate(model_classes)}

# Predict image
def get_prediction(image_bytes, topk=5):
    try:
        tensor = transform_image(image_bytes)
        out = model.forward(tensor)
    except Exception:
        return 0, 'error'
    ps = torch.exp(out)
    # Find the topk predictions
    topk, topclass = ps.topk(topk, dim=1)
    topk, topclass = topk.squeeze().detach().numpy(), topclass.squeeze().detach().numpy()
    # Extract the actual classes and probabilities
    # prob_dict = {}
    # for i in range(len(model_classes)):
    #   prob_dict[model_classes[topclass[i]]] = topk[i]
    return model_classes[topclass[0]]#prob_dict

# Instantiate the Flask app
app = Flask(__name__, static_url_path='/static')
@app.route('/about')
def render_about_page():
    return render_template('about.html')

@app.route('/')
def render_page():
    return render_template('index.html')

@app.route('/uploadajax',methods=['POST'])
def upload_file():
    """
    retrieve the image uploaded and make sure it is an image file
    """
    file = request.files['file']
    image_extensions=['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']
    if file.filename.split('.')[1] not in image_extensions:
        return jsonify('Please upload an appropriate image file')
    """
    Load the trained densenet model
    """
    model = get_model()
    """
    Load variables needed to detect human face/dog pil image for dog detection and breed prediction and
    numpy for face detection
    """
    img_bytes = file.read()
    prob_dict = get_prediction(image_bytes=img_bytes)
    dog_breed = predict_breed_transfer(pil_image,model_transfer)
    return jsonify ('The predicted image is:{}'.format(prob_dict))

if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))