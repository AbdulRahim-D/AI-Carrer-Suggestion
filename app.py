from flask import Flask, jsonify, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
with open('model/recommender.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for the homepage that renders the index.html file
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Assume the incoming JSON is like this:
        # { "interest": "tech", "skill": "coding" }
        user_input = data
        input_df = pd.DataFrame([user_input])

        # Prepare the data (using one-hot encoding)
        input_data = pd.get_dummies(input_df)

        # Ensure the input data has all columns the model expects
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict the career
        prediction = model.predict(input_data)

        # Return the prediction
        return jsonify({'predicted_career': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route for the favicon
@app.route('/favicon.ico')
def favicon():
    return "", 204  # No content, so browser won't make further requests

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
