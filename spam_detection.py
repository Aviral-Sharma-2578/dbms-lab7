from flask import Flask,render_template,request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])

def main():
    if request.method == 'POST':

        # Load the saved model
        model_filename = "spam_detection_model.pkl"
        loaded_model = joblib.load(model_filename)

        # Load the TF-IDF vectorizer used during training
        vectorizer_filename = "tfidf_vectorizer.pkl"
        tfidf_vectorizer = joblib.load(vectorizer_filename)

        # Input string to classify
        input_text = request.form.get("inp")

        # Preprocess the input string (remove stopwords, apply TF-IDF)
        preprocessed_input = tfidf_vectorizer.transform([input_text])

        # Predict using the loaded model
        predicted_label = loaded_model.predict(preprocessed_input)

        if predicted_label[0] == 1:
            return render_template('home.html',message="It's a spam e-mail 😒")
        elif predicted_label[0] == 0:
            return render_template('home.html',message="Not a spam e-mail 👍")
  
    return render_template('home.html')
