import torchvision.models
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import logging
from model.BMMTNet import BmmtNetV7

model = BmmtNetV7(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2, 3])
#model = torchvision.models.resnet18(pretrained=True)  # 数量对不上



app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif','tif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def MyPredict(image_path):

    # Your data transformation
    normal_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model.eval()
    # Load and transform the single image

    image = Image.open(image_path)
    input_image = normal_transforms(image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(input_image)

    # Get the predicted label
    _, predicted_label = torch.max(output, 1)
    ##print(f"Predicted label:  {label_list[predicted_label.item()]}")

    # return label_list[predicted_label.item()]

    return predicted_label.item()


@app.route('/')
def index():
    if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'],"A")):
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],"A"))
    if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'],"B")):
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],"B"))
    if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'],"out")):
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],"out"))
    return render_template('index.html')


@app.route('/upload', methods=["POST"])
def upload_file():
    print(request.files)
    if 'file1' not in request.files or 'file2' not in request.files:
        return redirect(request.url)

    file1 = request.files['file1']  # 拿到file
    file2 = request.files['file2']


    if file1.filename == ''or file2.filename == '':
        return redirect(request.url)
    use_images=[]
    if file1 and allowed_file(file1.filename):
        filename1 = secure_filename(file1.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],"A", filename1)
        file1.save(file_path)
        use_images.append(filename1)
    if file2 and allowed_file(file2.filename):
        filename2 = secure_filename(file2.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],"B", filename2)
        file2.save(file_path)
        use_images.append(filename2)
    result = MyPredict(os.path.join(app.config['UPLOAD_FOLDER'],"A",use_images[0]))

    result_path=os.path.join(app.config['UPLOAD_FOLDER'],"out",filename1)
    result.save(result_path)
    return render_template('index.html', filenames= use_images, result=result)


@app.route('/uploads/<filename>')  # uploads/L6_00f153d186dbf5fa02e0558e47537fde.jpg
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True,port=4888)