# Import necessary libraries
from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from score import score

app = Flask(__name__)

# Load the pre-trained SVM model
with open("svm_model.pkl", "rb") as model_file:
    b_model = pickle.load(model_file)

TRAIN_DATASET = pd.read_csv('train.csv')

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(TRAIN_DATASET['text'])

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        
        input_text = request.form.get("text")
        
        if not input_text:
            content = request.json
            if len(content) == 0:
                return "Record not found", 400
            if 'text' in content:
                input_text = content['text']
        
        # # Vectorize the input text
        # if input_text:
        #     prediction, propensity = score(input_text, b_model, vectorizer, 0.5)
        # else:
        
        prediction, propensity = score(input_text, b_model, vectorizer, 0.5)

        # Create a JSON response
        response_data = {
            "prediction": int(prediction),
            "propensity": float(propensity)  # Convert to float for JSON serialization
        }

        return jsonify(response_data)
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spam Classification</title>
            <!-- Add any other head elements (stylesheets, scripts, etc.) here -->
        </head>
        <body>
            <h1>Spam or Not?</h1>
            <form method="post">
                <textarea name="text" rows="4" cols="50" placeholder="Enter your text here"></textarea>
                <br>
                <input type="submit" value="Predict">
            </form>
        </body>
        </html>
        """

if __name__ == "__main__":
    app.run(debug=True)
