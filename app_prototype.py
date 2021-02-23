# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:29:07 2020

@author: ms101
"""

# bird classifier app
#imports
import streamlit as st
import numpy as np
import torch
from PIL import Image
from fastai.vision.all import load_learner#, Path
from pathlib import PosixPath
#import pathlib
#temp = pathlib.WindowsPath
#pathlib.WindowsPath = pathlib.PosixPath
#app
st.title("Bird Classifier")
st.write("""
You can use this app to classify birds from images.

Did you take a picture of a bird, but aren't sure which species it is?
No Problem! Just upload your image and let the classifier tell you!

*Note: This first version only knows birds from Europe (100 Species), classifiers for other continents will be implemented soon*
""")
uploaded_file = st.file_uploader("Choose an image...", type="jpg") #file upload

learn_inf = load_learner(PosixPath("EU_first_34.pkl")) #load trained model
learn_inf.model.cpu()
#classify
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Your Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = np.asarray(img)
    label = learn_inf.predict(image)
    if label[0][0] in "AEIOU":
        st.write("## This looks like an")
    else:
        st.write("## This looks like a")
    st.title(label[0].replace("_"," "))
