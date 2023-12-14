import tensorflow
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm
feature=np.array(pickle.load(open('feature.pkl','rb')))
filenames=np.array(pickle.load(open('files.pkl','rb')))
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()])
#print(model.summary())
st.title('Fashion Recommender System')
def uploadfile(uploadedfile):
    try:
        with open(os.path.join('uploads',uploadedfile.name),'wb') as f:
            f.write(uploadedfile.getbuffer())
        return 1
    except:
        return 0
def extractfeature(image_path,model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    expanimg = np.expand_dims(img_arr, axis=0)
    preproimg = preprocess_input(expanimg)
    result=model.predict(preproimg).flatten()
    result=result/norm(result)
    return result
def recommend(featured,feature):
    neigh=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neigh.fit(feature)
    dist,indice=neigh.kneighbors([featured])
    return indice
uploadedfile=st.file_uploader("Choose an image of item")
if uploadfile is not None:
    if uploadfile(uploadedfile):
        dispimg=Image.open(uploadedfile)
        st.image(dispimg)
        featured=extractfeature(os.path.join('uploads',uploadedfile.name),model)
        indice=recommend(featured,feature)
        c1,c2,c3,c4,c5=st.columns(5)
        with c1:
            st.image(filenames[indice[0][0]])
        with c2:
            st.image(filenames[indice[0][1]])
        with c3:
            st.image(filenames[indice[0][2]])
        with c4:
            st.image(filenames[indice[0][3]])
        with c5:
            st.image(filenames[indice[0][4]])
