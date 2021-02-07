import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from streamlit import caching
from streamlit_folium import folium_static

import app_body as body

warnings.filterwarnings('ignore')

st.title('Spotify Artist Evaluation - Lola Amour')

## Side Bar Information
image = Image.open('eskwelabs.png')
st.sidebar.image(image, caption='', use_column_width=True)
st.sidebar.markdown("<h1 style='text-align: center;margin-bottom:50px'>DS Cohort VI</h1>", unsafe_allow_html=True)

## Create Select Box and options
add_selectbox = st.sidebar.radio(
    "",
    ("Objectives", "Client Profile", "Data Information", "List of Tools", "EDA - About the Industry", "EDA - Audio Features", "Modeling - Results",
    "Recommendation Engine", "Conclusions and Recommendations", "Contributors")
)


if add_selectbox == 'Objectives':
    body.objectives()

elif add_selectbox == 'Client Profile':
    body.client_profile()

elif add_selectbox == 'Data Information':
    body.dataset()

elif add_selectbox == 'List of Tools':
    body.tools()

elif add_selectbox == 'EDA - About the Industry':
    body.eda1()

elif add_selectbox == 'EDA - Audio Features':
    body.eda2()

elif add_selectbox == 'Modeling - Results':
    body.modeling()

elif add_selectbox == 'Recommendation Engine':
    body.recommendation_engine()

elif add_selectbox == 'Conclusions and Recommendations':
    body.candr()

elif add_selectbox == 'Contributors':
    body.contributors()