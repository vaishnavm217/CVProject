from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from time import sleep
import numpy as np
import h5py
import json
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
# import redis
import pickle
import os
from tqdm import tqdm
PATH="resources"
base_model = VGG19(weights='imagenet')
model_vgg = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
model_filename = os.path.join(PATH,'model.json')
model_weights_filename = os.path.join(PATH,'model_weights.h5')
metadata = json.load(open(os.path.join(PATH,'data_prepro.json'), 'r'))
metadata['ix_to_word'] = {str(word):int(i) for i,word in metadata['ix_to_word'].items()}

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

def predict(image_path="kite.jpeg",question="What is flying in the sky?"):
    json_file = open(model_filename, 'r')
    #print image_path,question
    loaded_model_json = json_file.read()
    print "MODEL",loaded_model_json
    #img = load_image_array(image_path)
    #print "image:",img
    print "Reading Model..."
    model = model_from_json(loaded_model_json)
    print "Loading Weights from ",model_weights_filename
    #print "model weights:",open(model_weights_filename,"r").read()
    model.load_weights(model_weights_filename)
    print "Loaded Weights..."
    img = load_image_array(image_path)
    img_vector = model_vgg.predict(img)
    question_vector = get_ques_vector(question)
    print question_vector,img_vector
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    pred = model.predict([img_vector, question_vector])[0]
    top_pred = pred.argsort()[-5:][::-1]
    print [(metadata['ix_to_ans'][str(_)].title(), round(pred[_]*100.0,2)) for _ in top_pred]
    return [(metadata['ix_to_ans'][str(_)].title(), round(pred[_]*100.0,2)) for _ in top_pred]
    # print (preds)
predict("dog.jpg","What animal is this?")
def eval(annotations):
    res = []
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    print "Reading Model..."
    model = model_from_json(loaded_model_json)
    print "Loading Weights..."
    model.load_weights(model_weights_filename)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    s = 0
    len_data = len(annotations)
    batch_size = 16
    end = batch_size-1
    while not s>=len_data-1:
        batch_img = []
        batch_ques = []
        for path in annotations[s:end+1]:
            img = load_image_array(path["img_path"])
            #img_vector = model_vgg.predict(img)
            question_vector = get_ques_vector(path["question"])
            batch_img.append(img)
            batch_ques.append(question_vector)
        img_vector = np.concatenate(batch_img)
        question_vector = np.concatnate(batch_ques)
        pred = model.predict([img_vector, question_vector])
        print pred
        s+=end+1
        end+=batch_size
        if end>len_data-1:
            end = len_data-1
        #top_pred = pred.argsort()[-5:][::-1]
        #res.append({"answer":metadata['ix_to_ans'][str(top_pred[0])].title(),"question_id":path["ques_id"]})
    print "Result ready"
    json.dump(res,open("result.json"))
#eval(json.load(open("../VQA_LSTM_CNN/data/vqa_raw_test.json","r")))
