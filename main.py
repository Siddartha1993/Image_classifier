from flask import Flask, render_template, request, redirect,Response
from flask_cors import CORS
import tensorflow
import numpy as np
from keras.preprocessing import image
from PIL import Image  
import pandas as pd
import base64
from io import BytesIO
import os




app = Flask(__name__)
CORS(app)

new_model=''

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER="/ImageClassifierApp/static/uploadedImages/"
app.config['UPLOAD_FOLDER'] = BASE_DIR+UPLOAD_FOLDER

@app.before_first_request
def load_model():
    global new_model
    new_model=tensorflow.keras.models.load_model('classifier.h5')

@app.route('/')
def landingPage():
   return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def upload_file():
    global new_model
    global b64
    vals = ['PCB', 'Turbine','Pump']
    result_df=pd.DataFrame(columns=["uploaded Image","Prediction"])
    # results={}
    try:
        files = request.files.getlist('file')
        print("filessss----",files)
        for file in files:
            print("files----------->",file)
            print("model----------->",new_model)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            # img = image(i, target_size=(300, 300))
            with Image.open(file) as img: 
            # img=Image.open(file)
                img=img.resize((300,300))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            images = np.vstack([x])
            classes = new_model.predict(images, batch_size=10)
            # results["filename"]=i
            # results["pred"]=str(vals[np.argmax(classes)])
            # buffer = BytesIO()
            # img.save(buffer,format="JPEG")                  
            # myimage = buffer.getvalue()  
            # b64="data:image/jpeg;base64,"+base64.b64encode(myimage).decode("UTF-8")
            # b64=b64.replace("/","//")
            # print("base64",b64)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], i.filename))
            result_df = result_df.append({'uploaded Image' : f'<img src="http://127.0.0.1:5000/static/uploadedImages/{file.filename}" width="150" height="150">', "Prediction" : str(vals[np.argmax(classes)])},  
                ignore_index = True) 
            print("df----",result_df)
    except Exception as e:
        print(e)
    result_df.index=result_df.index+1
    return result_df.to_html(classes="table table-resizable", table_id="result", escape=False)


if __name__ == '__main__':
    app.run(debug=True)