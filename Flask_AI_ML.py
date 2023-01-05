from flask import Flask,  render_template, redirect, url_for, request, flash, abort
from wtforms.validators import InputRequired
from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os

app=Flask(__name__, template_folder='templates')
app.config['SECRET_KEY']="dnkvvians@#"
app.config['UPLOAD_FOLDER']='static\\files'
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg']
app.config['MAX_CONTENT_LENGTH']=32*1024*1024

app.type=""
app.c=""
app.dis=""
app.path=""

class UploadFileForm(FlaskForm):
    file=FileField("File", validators=[InputRequired()])
    submit=SubmitField("Upload File")

def check_size(path):
    with Image.open(path) as image:
        width,height=image.size
        if width < 224 or height < 224:
            return True

def pneu_pred(fp: str):
        model = load_model("pneu.h5", compile=False)
        img = image.load_img(fp, target_size=(224, 224))
        imagee = image.img_to_array(img)
        imagee = np.expand_dims(imagee, axis=0)
        img_data = preprocess_input(imagee)
        pred = model.predict(img_data)
        if pred[0][0] > pred[0][1]:
            return False
        return True
def sc_pred(path):
    model=load_model('skin_cancer_detector_by_irfan.h5')
    i=image.load_img(path,target_size=(32,32))
    im=image.img_to_array(i)
    img=np.expand_dims(im,axis=0)
    prediction=model.predict(img)
    x=list(prediction[0])
    if max(x) == 0:
        return "Negative"
    if x.index(max(x))==0:
        return 'Actinic keratoses'
    elif x.index(max(x))==1:
        return 'Basal cell carcinoma'
    elif x.index(max(x))==2:
        return 'benign keratoses-like lesions'
    elif x.index(max(x))==3:
        return 'Dermatofibroma'
    elif x.index(max(x))==4:
        return 'Melanocytic nevi'
    elif x.index(max(x))==5:
        return 'Pyrogenic granulomas and hemorrhage'
    elif x.index(max(x))==6:
        return 'Melanoma'

@app.route('/')
def main_index():
    return render_template('index.html')

@app.route('/<pneupage>')
def pneu_page(pneupage):
    if pneupage =='DDAI.html':
        return redirect(url_for('pneu_page_to'))
    if pneupage == 'index.html':
        return redirect(url_for('main_index'))
    if pneupage == "DDAISC.html":
        return redirect(url_for('skin_cancer'))
    

@app.route('/DDAI.html', methods=["GET","POST"])
def pneu_page_to():
    error=""
    app.dis="Pnue"
    form=UploadFileForm()                                                                                
    if form.validate_on_submit():                                                                        
        file=form.file.data
        file_ext = os.path.splitext(file.filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            error="Files with only .jpg and .jpeg extensions are allowed"
        else:
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
            app.c=file.filename
            minpath=".\\static\\files\\" + app.c
            app.path=os.path.abspath(minpath)
            if check_size(app.path):
                error="The dimensions of the image must be greater than 224 px"
            else:
                return redirect(url_for('result'))
                app.type="Detector"
    return render_template('DDAI.html', form=form, error=error)

@app.route('/DDAISC.html', methods=["GET","POST"])
def skin_cancer():
    error=""
    app.dis="SC"
    form=UploadFileForm()                                                                                
    if form.validate_on_submit():                                                                        
        file=form.file.data   
        file_ext = os.path.splitext(file.filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            error="Files with only .jpg and .jpeg extensions are allowed"
        else:
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
            app.c=file.filename
            minpath=".\\static\\files\\" + app.c
            app.path=os.path.abspath(minpath)
            if check_size(app.path):
                error="The dimensions of the image must be greater than (224,224)"
            else:
                return redirect(url_for('result'))
            
    return render_template("DDAISC.html", form=form, error=error)

@app.route('/result.html', methods=["GET","POST"])
def result():    
    out=""
    minpath=".\\static\\files\\" + app.c
    app.path=os.path.abspath(minpath)
    if app.dis=="Pnue":
        app.dis="Pneumonia"
        typer="Detection"
        if pneu_pred(app.path):
            out=" Positive"
        else:
            out=" Negative"
    elif app.dis=="SC":
        app.dis="Skin Cancer"
        typer="Classifier"
        scfunc=sc_pred(app.path)
        if scfunc=="Negative":
            out=" Negative"
        else:
            out=" Positive; Type: " + scfunc
    return render_template('result.html',filename=app.c, disease=app.dis, pneu_result=out, type=typer)

if __name__=="__main__":
    app.run(debug=True)
    

