from flask import Flask, request, jsonify
import base64
from io import BytesIO
import re
from PIL import Image, ImageFilter
from flask import render_template

from model import *

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index(name=None):
    if request.method == 'POST':
        img_url   = request.values['imageBase64']

        image_data = re.sub('^data:image/.+;base64,', '', img_url)
        image_decoded = base64.b64decode(image_data)
        image_io = BytesIO(image_decoded)
        img = Image                                \
                .open(image_io)                    \
                .convert("RGB")                    \
                .resize((45, 45), Image.ANTIALIAS) \
                .filter(ImageFilter.SHARPEN)       \
                .save("test.png")

        img = Image.open("test.png")
        pixels = img.load()

        for i in range(img.size[0]): # for every pixel:
            for j in range(img.size[1]):
                if pixels[i,j] < (250, 250, 250):
                    pixels[i,j] = (0, 0, 0)

        img.save("test.png")

        loaded_model = Net()
        loaded_model.load_state_dict(torch.load("saved_model.pt", map_location="cpu"))
        loaded_model.eval()
        loaded_model.to(device)

        loaded_image = image_loader("test.png")
        output = loaded_model(loaded_image)
        _, predicted = torch.max(output.data, 1)

        pred = class_names[predicted]

        return jsonify(pred)
        #return render_template("test.html", response=pred )


    return render_template("test.html", name=name)
