# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:50:07 2021 (latest version)

@author: Merlin Schäfer
"""

# bird classifier app
#imports
import streamlit as st
import numpy as np
import torch
from PIL import Image
from fastai.vision.all import load_learner, Path
#from pathlib import Path
import pathlib
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath
#app
st.title("Bird Classifier")
st.write("""
You can use this app to classify birds from images.

Did you take a picture of a bird, but aren't sure which species it is?
No Problem! Just upload your image and let the classifier tell you!

*Note: This first version only knows birds from Europe (100 Species) and North America (224 Species), classifiers for other continents will be implemented soon*
""")
uploaded_file = st.file_uploader("Choose an image...", type="jpg") #file upload
continents = {"Europe": "EU",
"North America":"NA", 
"South America":"SA", 
"Africa":"AF", 
"Asia":"AS", 
"Australia Oceania":"AU", 
"Antarctica":"AN"}

habitat = st.sidebar.selectbox("On which continent did you spot this bird?", list(continents.keys()))

#load model
try:
    learn_inf = load_learner(Path(f"C:/Users/ms101/OneDrive/DataScience_ML/projects/birds_classifier/{continents.get(habitat)}_first_34.pkl"))#load trained model
    learn_inf.model.cpu()
except FileNotFoundError:
    st.title("There seems to be no classifier for this habitat, please choose another one")
#classify
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Your Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = np.asarray(img)
    label,prob_idx,prob = learn_inf.predict(image)
    prob_perc = round(float(prob[prob_idx])*100,2)
    if label[0] in "AEIOU":
        st.write("## This looks like an")
    else:
        st.write("## This looks like a")
    st.title(label.replace("_"," "))
    st.write(f"The classifier is {prob_perc}% sure")
    st.write("""*Note: if the classifier is not at least 90 percent sure, 
    there is a good chance that it either does not know the species 
    or the picture is not of a high enough quality*""")



