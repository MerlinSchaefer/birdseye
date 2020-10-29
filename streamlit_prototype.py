# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:29:07 2020

@author: ms101
"""

# bird classifier app
#imports
import streamlit as st
import torch
import numpy as np
from PIL import Image
#app
st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg") #file upload

learn_inf = torch.load("first_training.pkl") #load trained model
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
