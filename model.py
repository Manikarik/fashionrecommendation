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
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()])
#print(model.summary())
fealist=np.array(pickle.load(open('feature.pkl','rb')))
filename=np.array(pickle.load(open('filename.pkl','rb')))
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
filenames=[]
for f in os.listdir('images'):
    filenames.append(os.path.join('images',f))
#print(len(filenames))
feature=[]
for f in tqdm(filenames):
    feature.append(extractfeature(f,model))
pickle.dump(feature,open('feature.pkl','wb'))
pickle.dump(filenames,open('files.pkl','wb'))
