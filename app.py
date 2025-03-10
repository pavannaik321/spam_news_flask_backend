# import streamlit as st
# import pickle
# import re
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# port_stem = PorterStemmer()
# vectorization = TfidfVectorizer()

# vector_form = pickle.load(open('vector.pkl', 'rb'))
# load_model = pickle.load(open('model.pkl', 'rb'))

# def stemming(content):
#     con=re.sub('[^a-zA-Z]', ' ', content)
#     con=con.lower()
#     con=con.split()
#     con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
#     con=' '.join(con)
#     return con

# def fake_news(news):
#     news=stemming(news)
#     input_data=[news]
#     vector_form1=vector_form.transform(input_data)
#     prediction = load_model.predict(vector_form1)
#     return prediction



# if __name__ == '__main__':
#     st.title('Fake News Classification app ')
#     st.subheader("Input the News content below")
#     sentence = st.text_area("Enter your news content here", "",height=200)
#     predict_btt = st.button("predict")
#     if predict_btt:
#         prediction_class=fake_news(sentence)
#         print(prediction_class)
#         if prediction_class == [0]:
#             st.success('Reliable')
#         if prediction_class == [1]:
#             st.warning('Unreliable')






import pickle
import re
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')

# Load trained model
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# NLP Preprocessing
port_stem = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    content = ' '.join([port_stem.stem(word) for word in content if word not in stopwords.words('english')])
    return content

# FastAPI Setup
app = FastAPI()

# CORS (Allows frontend requests)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Input Schema
class NewsRequest(BaseModel):
    news: str

@app.post("/predict")
def predict_fake_news(request: NewsRequest):
    processed_news = stemming(request.news)
    vector_input = vector_form.transform([processed_news])
    prediction = load_model.predict(vector_input)
    return {"prediction": int(prediction[0])}

















# import streamlit as st
# import pickle
# import re
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load models
# port_stem = PorterStemmer()
# vectorization = TfidfVectorizer()
# vector_form = pickle.load(open('vector.pkl', 'rb'))
# load_model = pickle.load(open('model.pkl', 'rb'))

# # Preprocess function
# def stemming(content):
#     con = re.sub('[^a-zA-Z]', ' ', content)
#     con = con.lower().split()
#     con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
#     return ' '.join(con)

# # Prediction function
# def fake_news(news):
#     news = stemming(news)
#     input_data = [news]
#     vector_form1 = vector_form.transform(input_data)
#     prediction = load_model.predict(vector_form1)
#     return prediction

# # Streamlit UI
# st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# # Apply custom styling
# st.markdown(
#     """
#     <style>
#         body {
#             background-color: #f4f4f4;
#         }
#         .main-title {
#             font-size: 36px;
#             font-weight: bold;
#             text-align: center;
#             color: #ff4b4b;
#         }
#         .subheader {
#             text-align: center;
#             color: #333;
#         }
#         .report-box {
#             background-color: white;
#             padding: 15px;
#             border-radius: 10px;
#             box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
#         }
#         .stButton>button {
#             background: linear-gradient(90deg, #ff4b4b, #ff7b7b);
#             color: white;
#             border: none;
#             padding: 10px;
#             font-size: 16px;
#             border-radius: 8px;
#             width: 100%;
#             transition: 0.3s ease;
#         }
#         .stButton>button:hover {
#             background: linear-gradient(90deg, #e63e3e, #ff4b4b);
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown('<h1 class="main-title">📰 Fake News Detection App</h1>', unsafe_allow_html=True)
# st.markdown('<h3 class="subheader">Enter the news content below to check its reliability.</h3>', unsafe_allow_html=True)

# sentence = st.text_area("✍️ Paste News Content Here:", "", height=200)

# if st.button("🔍 Predict"):
#     prediction_class = fake_news(sentence)
    
#     if prediction_class == [0]:
#         st.success("✅ This news is **Reliable**.")
#     elif prediction_class == [1]:
#         st.warning("⚠️ This news is **Unreliable (Fake)**.")
    
# st.markdown("</div>", unsafe_allow_html=True)
