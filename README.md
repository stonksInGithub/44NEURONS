# DDAI-html
Disease detection/classification using neural network and tensorflow in python
<br>
This is the source code for the project(named "**44NEURONS**").
<br>
Build with 💖 using python!

## Back-end & Front-end


We used flask to integrate the front-end and back-end
Python modules required:
```sh
python3 -m pip --upgrade install flask, tensorflow, keras_preprocessing matplotlib
```

## Running Locally
**Note**: Since github uploads are restricted to 100mb the model file for pneumonia detector can be downloaded through this [drive link](https://drive.google.com/file/d/1RMY8TB4TUt872VeT7or2Y9YcSX714jQc/view?usp=share_link).<br>
After installing the specified modules and the dataset and the `.h5` files. Extract the dataset and copy the ```test, train, val``` folders to the project folder.
Change the folder path in Flask_AI_ML.py file and `UPLOAD_FOLDER` path too.
Run the flask app py:
```sh
# For Linux
python3 Flask_AI_ML.py

# For windows
py3 Flask_AI_ML.py
```

The source code is licensed under MIT and further use of this code, credit should be given.

## Datasets used
[Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/code)
<br>
[Skin cancer dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
