from flask import Flask,render_template,request,jsonify

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pickle
import joblib
import pandas as pd
import numpy as np
import os


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)
CORS(app)

try:
    disease_model = joblib.load('disease_prediction_model.joblib')
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    print("Disease prediction model loaded successfully!")
except Exception as e:
    print(f"Error loading disease prediction model: {e}")
    disease_model = None
    label_encoder = None

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):
    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
@app.route('/prediction', methods=['GET', 'POST'])
def predict_disease():
    try:
        if disease_model is None:
            return jsonify({
                'success': False,
                'error': 'Disease prediction model is not loaded.',
                'disease': 'Model Error',
                'confidence': 0,
                'recommendation': ['Please check your setup.']
            })
    
    # Get JSON data from the request
        data = request.get_json()

# Extract symptoms from the request
        symptoms = [
            data.get('appetite', 0),
            data.get('energy', 0),
            data.get('digestive', 0),
            data.get('breathing', 0),
            data.get('movement', 0),
            data.get('temperature', 0)
        ]

        # Convert to numpy array and reshape for prediction
        symptoms_array = np.array(symptoms).reshape(1, -1)

        # Make prediction
        prediction = disease_model.predict(symptoms_array)

        # Get prediction probabilities for confidence
        if hasattr(disease_model, 'predict_proba'):
            probabilities = disease_model.predict_proba(symptoms_array)
            confidence = np.max(probabilities) * 100
        else:
            confidence = 85

            # Decode the prediction if you used label encoding
        if label_encoder:
            disease_name = label_encoder.inverse_transform(prediction)[0]
        else:
            disease_name = prediction[0]

            # Generate recommendations
        recommendations = get_disease_recommendations(disease_name, symptoms)
        return jsonify({
            'success': True,
            'disease': disease_name,
            'confidence': round(confidence, 1),
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'disease': 'Prediction Error',
            'confidence': 0,
            'recommendations': ['Please try again or consult a veterinarian']
        })
    
    # NEW: Disease recommendations function
def get_disease_recommendations(disease, symptoms):
    recommendations_dict = {
        'Healthy': [
            'Your pet appears to be in good health',
            'Continue regular feeding schedule',
            'Maintain exercise routine',
            'Schedule regular checkups'
        ],
        'Digestive Issues': [
            'Monitor food and water intake closely',
            'Consider bland diet temporarily',
            'Ensure adequate hydration',
            'Contact veterinarian if symptoms persist'
        ],
        'Respiratory Problems': [
            'Ensure good air circulation',
            'Avoid smoke and strong odors',
            'Monitor breathing patterns',
            'Seek immediate vet care if breathing worsens'
        ]
        # Add more disease-specific recommendations
    }
    
    return recommendations_dict.get(disease, [
        'Monitor your pet closely',
        'Maintain regular care routine',
        'Contact veterinarian for proper diagnosis',
        'Follow up if symptoms change'
    ])

# Serve your HTML files
@app.route('/')
def index():
    return send_from_directory('.', 'chatbot.html')

@app.route('/disease')
def disease_page():
    return send_from_directory('.', 'disease.html')

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)

if __name__ == "__main__":
    app.run()

