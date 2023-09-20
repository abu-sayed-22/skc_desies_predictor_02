"""
ONly test purpose

"""


import os
from flask import Flask, request, render_template, redirect, url_for, make_response
import base64
# PILfor Image
from PIL import Image
from io import BytesIO
import json
# loiading the learner
from fastcore.all import *
from fastai.learner import load_learner
from flask_cors import CORS
app = Flask(__name__)


# setting up the cors for ignorring the cors policy
CORS(app)


@app.get('/')
def home():
    return {
        "wlc": "something is gonna update soon"
    }


# defining the predictroue
@app.post('/predict')
def predict():
    """
    things that we should recive
    {
        "image":"base64code",
        "apikey":"NOthing"
    }

    """
    try:
        # loading the data from the user and get image base64 string
        data = json.loads(request.data)
        base64_image_data = data["image"]
        apikey = data["apikey"]

        # checking the api key so that false request cannot be send
        if apikey!="sayedSKC@386":
            return make_response({"error": "401 ", "message": "Unauthorized "}, 401)





        # at first convert the base64 image into image
        try:
            binary_data = base64.b64decode(base64_image_data)
            # image = Image.open(BytesIO(binary_data))
            with open("input_img.jpg", 'wb') as img:
                img.write(binary_data)
            # image.show()
        except (base64.binascii.Error, OSError, Exception) as e:

            return make_response({"error": "500", "message": "Conversion is not poossible"}, 500)

        # now fit to the model and return the result to the user
        learner = load_learner("./export.pkl")
        # result =learner.predict(image)
        pred_class, pred_idx, proabs = learner.predict('input_img.jpg')
        # deleting the image file for cleaning the gubage
        os.unlink('./input_img.jpg')
        proab_labels = ['potato_normal', 'potato_late_blight',
                        'totamto_normal', 'totamto_late_blight']
        proabs = proabs.numpy().tolist()
        
        return {
            "output": {
                "pred_class": pred_class,
                "proab_labels": proab_labels,
                "proabs": proabs

            }
        }
    except:
        # handeling any else case
        return  make_response({"status": 500, "message": "internal server error"}, 500)




# Setting up the default 404 error handler
@app.errorhandler(404)
def page_not_found(error):
    return make_response({"status": 404, "message": "page not found"}, 404)


# finally run the app
if __name__ == "__main__":
    app.run(debug=False)
