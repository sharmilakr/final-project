import streamlit as st
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import easyocr
from PIL import Image
import pandas as pd
import numpy as np
import re
import mysql.connector
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
import easyocr
from joblib import load
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
# SETTING PAGE CONFIGURATIONS
#icon = Image.open("image1.jpg")
st.set_page_config(page_title="Final Project:| By Sharmila ",
                   #page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This web application is created to the model prediction, price prediction, Image processing and NLP *!"""})
st.markdown("<h1 style='text-align: center; color: Green;",
            unsafe_allow_html=True)

#st.snow
#python -m streamlit run pravi_st.py


# CREATING OPTION MENU
  
selected = option_menu(None, ["Home", "Customer_conversion","EDA", "Product_recommendation","NLP","Image"],
                       icons=["house", "cloud-upload", "pencil-square"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "25px", "text-align": "centre", "margin": "-3px",
                                            "--hover-color": "#545454"},
                               "icon": {"font-size": "30px"},
                               "container": {"max-width": "5000px"},
                               "nav-link-selected": {"background-color": "#ff5757"}})

# HOME MENU
if selected == "Home":
    col1, col2 = st.columns(2)
    with col1:
        
        st.markdown("## :green[**Technologies Used :**] Machine Learning,Python,easy OCR, Streamlit, Pandas")
    with col2:
        st.write(
            '## This project is the combination of Machine Learning models, NLP, Complete EDA process and Image processing ')


# HOME MENU
if selected == "Home":
    col1, col2 = st.columns(2)
    with col1:
        #st.image(Image.open("image2.png"), width=500)
        st.markdown("## :green[**Technologies Used :**] Machine Learning,Python,easy OCR, Streamlit,Pandas")
    with col2:
        st.write('## This project is the comination of Machine Learning models, NLP, Complete EDA process and Image processing ')

# Customer conversion
data = pd.read_csv("classification_data.csv")

# Load the models inside the Streamlit app
if selected == "Customer_conversion":
    col1, col2 = st.columns(2)
    with col1:
        with open("decision_tree_model.pkl", 'rb') as file:
            decision_tree_model = pickle.load(file)

                
    channelgrouplist = list(data['channelGrouping'].unique())
    channelgrouplist.sort()
    devices = list(data['device_deviceCategory'].unique())
    devices.sort()
    regions = list(data['geoNetwork_region'].unique())
    regions.sort()
    sources = list(data['latest_source'].unique())
    sources.sort()
    keyword = list(data['latest_keyword'].unique())
    keyword.sort()
    product_arr = list(data['products_array'].unique())
    product_arr.sort()

    device_deviceCategory = st.selectbox("Select Device Category", devices)
    geoNetwork_region = st.selectbox("Select GeoNetwork Region", regions)
    historic_session = st.number_input("Enter historic_session", min_value=0, value=0)
    historic_session_page = st.number_input("Enter historic_session_page", min_value=0, value=0)
    avg_session_time = st.number_input("Enter avg_session_time", min_value=0, value=0)
    avg_session_time_page = st.number_input("Enter avg_session_time_page", min_value=0, value=0)
    single_page_rate = st.number_input("Enter single_page_rate", min_value=0, value=0)
    sessionQualityDim = st.number_input("Enter sessionQualityDim", min_value=0, value=0)
    latest_visit_id = st.number_input("Enter latest_visit_id", min_value=0, value=0)
    latest_visit_number = st.number_input("Enter latest_visit_number", min_value=0, value=0)
    time_latest_visit = st.number_input("Enter time_latest_visit", min_value=0, value=0)
    avg_visit_time = st.number_input("Enter avg_visit_time", min_value=0, value=0)
    visits_per_day = st.number_input("Enter visits_per_day", min_value=0, value=0)
    latest_source = st.selectbox("Select Latest Source",sources)
    latest_medium = st.selectbox("Select Latest Medium", data['latest_medium'].unique())
    latest_keyword = st.selectbox("Enter Latest Keyword",keyword)
    latest_isTrueDirect = st.checkbox("Is True Direct", value=False)
    time_on_site = st.number_input("Enter time_on_site", min_value=0, value=0)
    products_array = st.selectbox("Enter product array",product_arr)
    transactionRevenue = st.number_input("Enter transactionRevenue", min_value=0, value=0)
    count_hit = st.number_input("Enter counthit")
    channelGrouping = st.selectbox("Enter channelGrouping",channelgrouplist)

    channels = int(channelgrouplist.index(channelGrouping))
    device = int(devices.index(device_deviceCategory))
    region = int(regions.index(geoNetwork_region))
    source = int(sources.index(latest_source))
    keywords = int(keyword.index(latest_keyword))
    product_arrr = int(product_arr.index(products_array))

    additional_feature = {
        "count_hit": count_hit,
        'channelGrouping': channelgrouplist.index(channelGrouping),
        'device_deviceCategory': devices.index(device_deviceCategory),
        'geoNetwork_region': regions.index(geoNetwork_region),
        'historic_session':historic_session,
        'historic_session_page':historic_session_page,
        'avg_session_time':avg_session_time,
        'avg_session_time_page':avg_session_time_page,
        'single_page_rate':single_page_rate,
        'sessionQualityDim':sessionQualityDim,
        'latest_visit_id':latest_visit_id,
        'latest_visit_number':latest_visit_number,
        'time_latest_visit':time_latest_visit,
        'avg_visit_time':avg_visit_time,
        'visits_per_day':visits_per_day,
        'latest_source':source,
        'latest_keyword':keywords,
        'latest_isTrueDirect':latest_isTrueDirect,
        'time_on_site':time_on_site,
        'transactionRevenue':transactionRevenue,
        'products_array':product_arrr
    }
    

    all_features = [
        'count_hit', 'channelGrouping', 'device_deviceCategory',
        'geoNetwork_region', 'historic_session', 'historic_session_page',
        'avg_session_time', 'avg_session_time_page', 'single_page_rate',
        'sessionQualityDim', 'latest_visit_id', 'latest_visit_number',
        'time_latest_visit', 'avg_visit_time', 'visits_per_day',
        'latest_source', 'latest_keyword', 'latest_isTrueDirect',
        'time_on_site', 'transactionRevenue', 'products_array',
    ]

    if st.button("Predict Conversion"):
        df = pd.DataFrame([additional_feature])
        
        st.dataframe(df)
        dtm=decision_tree_model.predict(df)
        
        st.write(dtm)
        if dtm[0]==0:
            st.write("Not converted")
        else:
            st.write("Converted")

    with col2:
        score_test = {
    "Model":["Random Forest","AdaBoost ","Gradient Boosting","XGBoost","Bagging","Decision Tree","Logistic_regression","Knn"],
    "Accuracy":[0.99,0.97,0.98,0.99,0.99,0.96,0.74,0.52],
    "Precision":[0.99,0.98,0.98,0.99,0.99,0.96,0.72,0.52],
    "Recall":[0.99,0.97,0.99,0.99,0.99,0.98,0.82,1],
    "F1 Score":[0.99,0.97,0.99,0.99,0.99,0.97,0.77,0.68]
        }
    dff=pd.DataFrame(score_test)

    st.dataframe(dff)



#IMAGE PROCESSING
        
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import easyocr
import numpy as np
# Image processing

if selected == "Image":
    
    #st.set_page_config(page_title="Image Processing App", page_icon=":camera:", layout="wide")

# Function to perform image processing
    def process_image(image, resize_width, resize_height, conversion_mode,
                    enhance_contrast, enhance_brightness, gaussian_blur_radius, sharpness_factor):
        # Resize the image
        resized_image = image.resize((resize_width, resize_height))

        # Convert the image to the selected color mode
        converted_image = resized_image.convert(conversion_mode)

        # Perform image transformations based on user input
        grey_scale = image.convert('L')
        resized_grey_image = grey_scale.resize((500, 300))
        blur_image = resized_grey_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius))
        contrast_image = ImageEnhance.Contrast(image).enhance(enhance_contrast)
        brightness_image = ImageEnhance.Brightness(image).enhance(enhance_brightness)
        sharpness_image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

        return resized_image, converted_image, resized_grey_image, blur_image, contrast_image, brightness_image, sharpness_image

# Main function
    def main():
        # Get user input for image upload
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        # Check if an image was uploaded
        if uploaded_image is not None:
            # Open the uploaded image
            image_u = Image.open(uploaded_image)

            # Display the uploaded image
            st.image(image_u, caption='Uploaded Image', use_column_width=True)

            # Get user input for resizing
            resize_width = st.slider("Resize Width", 50, 1000, 500)
            resize_height = st.slider("Resize Height", 50, 1000, 300)

            # Get user input for image conversion
            conversion_mode = st.selectbox("Select Color Mode", ["RGB", "RGBA", "L", "CMYK"])

            # Get user input for image transformation parameters
            enhance_contrast = st.slider("Enhance Contrast", 0, 300, 100)
            enhance_brightness = st.slider("Enhance Brightness", 0.0, 5.0, 1.0)
            gaussian_blur_radius = st.slider("Gaussian Blur Radius", 0, 10, 2)
            sharpness_factor = st.slider("Sharpness Factor", 0.0, 10.0, 1.0)

        # Process the image
        resized_image, converted_image, resized_grey_image,  \
        contrast_image, brightness_image,blur_image, sharpness_image = process_image(
            image_u, resize_width, resize_height, conversion_mode,
            enhance_contrast, enhance_brightness, gaussian_blur_radius, sharpness_factor
        )

        # Display transformed images
        st.image(resized_image, caption=f'Resized Image ({resize_width}x{resize_height})', use_column_width=True)
        st.image(converted_image, caption=f'Converted Image (Mode: {conversion_mode})', use_column_width=True)
        st.image(resized_grey_image, caption='Grayscale Image', use_column_width=True)
        st.image(blur_image, caption='Blurred Image', use_column_width=True)
        st.image(contrast_image, caption='Contrast Enhanced Image', use_column_width=True)
        st.image(brightness_image, caption='Brightness Enhanced Image', use_column_width=True)
        st.image(sharpness_image, caption='Sharpened Image', use_column_width=True)

        # Write image processing summary
        st.write("## Image Processing Summary")
        st.write(f"Image Resized to: {resize_width}x{resize_height}")
        st.write(f"Converted to Color Mode: {conversion_mode}")
        st.write(f"Contrast Enhancement Factor: {enhance_contrast}")
        st.write(f"Brightness Enhancement Factor: {enhance_brightness}")
        st.write(f"Gaussian Blur Radius: {gaussian_blur_radius}")
        st.write(f"Sharpness Enhancement Factor: {sharpness_factor}")

# Run the main function
    if __name__ == "__main__":
        main()

#NLP
        
# Required Libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to perform NLP pre-processing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return lemmatized_tokens

# Function to find keywords
def find_keywords(tokens):
    frequency_dist = nltk.FreqDist(tokens)
    keywords = frequency_dist.most_common(5) # Get top 5 most common words
    return keywords

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"
    return sentiment, sentiment_score

def main():
    st.title("Text Analysis with Streamlit")

    # Input text from user
    text = st.text_area("Enter your text here:")

    # Perform NLP pre-processing
    if text:
        st.header("NLP Pre-processing:")
        
        # Convert to lowercase
        text_lower = text.lower()
        st.subheader("Converting to Lowercase:")
        st.write(text_lower)
        
        # Tokenization, Removing Stopwords, and Lemmatization
        tokens = preprocess_text(text)
        st.subheader("Tokenization and Removing Stopwords:")
        st.write(tokens)

        # Bag of Words & Word Frequencies
        st.header("Bag of Words & Word Frequencies:")
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(tokens)
        word_frequencies = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
        st.write("Word Frequencies:")
        st.write(word_frequencies)

        # Hierarchical Clustering
        st.header("Hierarchical Clustering:")
        clustering = AgglomerativeClustering(n_clusters=5)
        cluster_labels = clustering.fit_predict(X.toarray())
        st.write("Cluster Labels:")
        st.write(cluster_labels)

        # Word Embeddings
        st.header("Word Embeddings:")
        model = Word2Vec([tokens], min_count=1)
        word_embeddings = {word: model.wv[word].tolist() for word in model.wv.index_to_key if word in model.wv}
        st.write("Word Embeddings for 'love':")
        if 'love' in word_embeddings:
            st.write(word_embeddings['love'])
        else:
            st.write("Word 'love' not found in vocabulary.")

        
        # Find keywords
        st.header("Keywords:")
        keywords = find_keywords(tokens)
        st.write(keywords)

    # Perform sentiment analysis
        st.header("Sentiment Analysis:")
        sentiment, sentiment_score = analyze_sentiment(text)
        st.write("Sentiment:", sentiment)
        st.write("Sentiment Score:", sentiment_score)

if __name__ == "__main__":
    main()

#PRODUCT RECOMMENDATION

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load the product catalog
df = pd.read_csv('C:/Users/manik/Desktop/FINAL PROJECT/myntra_products_catalog.csv')

# Fill any missing values with empty string
df['Description'] = df['Description'].fillna('')

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['Description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reverse mapping of indices and product names
indices = pd.Series(df.index, index=df['ProductName']).drop_duplicates()

# Function to get the top N similar products
def get_similar_products(product_name, n=5):
    idx = indices[product_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    product_indices = [i[0] for i in sim_scores]
    return df['ProductName'].iloc[product_indices]

# Create a Streamlit app to display the recommendations
st.title('Product Recommendation System')
product_names = df['ProductName'].tolist()
product_name = st.multiselect('Enter a product name:', product_names, format_func=lambda x: x)
if product_name:
    similar_products = get_similar_products(product_name[0])
    st.write('Top 5 similar products to', product_name[0], ':')
    for i, product in enumerate(similar_products):
        st.write(i+1, '.', product)