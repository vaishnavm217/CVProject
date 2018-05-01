from flask import Flask
app = Flask(__name__,static_folder="static/")
import os
from flask import jsonify
from flask import render_template,request
# from VQA import predict
import sys
import random
import string
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model,model_from_json
import json
import numpy as np
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences

UPLOAD_FOLDER = 'static'
PATH=None
base_model = None
model_vgg = None
model_filename = None
model_weights_filename = None
metadata = None
model = None
@app.route("/", methods = ['GET', 'POST'])
def home():
    print "Hello"
    print request.args,request.form
    if request.method == 'POST':
        # print request.args,request.form
        if "question" in request.form:
            print "POST FORM"
            file1 = request.files['file']
            ques = request.form["question"]
            temp = os.path.join(UPLOAD_FOLDER, ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])+'.jpg')
            file1.save(temp)
            result = predict(temp,ques)
            os.remove(temp)
            return render_template('index.html',result=result,success=True,im=temp)
            # return jsonify(result=result,success=True)
        else:
            print "NO question came"
            return "oops"
    else:
        return render_template('index.html')

def load_image_array(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_ques_vector(question):
    question_vector = []
    seq_length = 26
    word_index = metadata['ix_to_word']
    for word in word_tokenize(question.lower()):
        if word in word_index:
            question_vector.append(word_index[word])
        else:
            question_vector.append(0)
    question_vector = np.array(pad_sequences([question_vector], maxlen=seq_length))[0]
    question_vector = question_vector.reshape((1,seq_length))
    return question_vector

def load_model():
    global PATH,base_model,model_vgg,model_filename,model_weights_filename,metadata,model
    PATH="resources"
    base_model = VGG19(weights='imagenet')
    model_vgg = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    model_filename = os.path.join(PATH,'model.json')
    model_weights_filename = os.path.join(PATH,'model_weights.h5')
    metadata = json.load(open(os.path.join(PATH,'data_prepro.json'), 'r'))
    metadata['ix_to_word'] = {str(word):int(i) for i,word in metadata['ix_to_word'].items()}
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_filename)
    predict("dog.jpg","what animal is this?")

def predict(image_path="kite.jpeg",question="What is flying in the sky?"):
    img = load_image_array(image_path)
    print(image_path)
    img_vector = model_vgg.predict(img)
    question_vector = get_ques_vector(question)
    print question_vector,img_vector
    pred = model.predict([img_vector, question_vector])[0]
    top_pred = pred.argsort()[-5:][::-1]
    print [(metadata['ix_to_ans'][str(_)].title(), round(pred[_]*100.0,2)) for _ in top_pred]
    return [(metadata['ix_to_ans'][str(_)].title(), round(pred[_]*100.0,2)) for _ in top_pred]

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0',debug=False)
sys.stdout.flush()
