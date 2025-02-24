import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from flask import Flask, request, jsonify
import io
from PIL import Image
import numpy as np

#simple CNN model
class MNISTCNNModel(nn.Module):
    def __init__(self):
        super(MNISTCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

#load model
model = MNISTCNNModel()
model.load_state_dict(torch.load('best_mnist_model_cs791_CNN (1).pth'))

model.eval()

#flask app
app = Flask(__name__)

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)   

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_byte = file.read()
    input_tensor = transform_image(img_byte)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
    
    return jsonify({'prediction': prediction})

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>MNIST Predictor</title>
        </head>
        <body>
            <h1>MNIST Digit Predictor</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=9000)