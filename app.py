#importing required libraries

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
from convert import convertion
warnings.filterwarnings('ignore')
from feature import FeatureExtraction
import google.generativeai as genai
import traceback

# Configure Gemini API
genai.configure(api_key='AIzaSyCdS4m_DBfscAtECUwR2kIYX4IN4JYEDtE')  # Configured with the provided API key
model = genai.GenerativeModel('gemini-1.5-flash')

file = open("newmodel.pkl","rb")
gbc = pickle.load(file)
file.close()

app = Flask(__name__)
#from flask import Flask, render_template, request
@app.route("/")
def home():
    return render_template("index.html")
@app.route('/result',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        url = request.form["name"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30)
    
        y_pred =gbc.predict(x)[0]
            #1 is safe
            #-1 is unsafe
        #y_pro_phishing = gbc.predict_proba(x)[0,0]
        #y_pro_non_phishing = gbc.predict_proba(x)[0,1]
            # if(y_pred ==1 ):
        #3pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        #xx =y_pred
        name=convertion(url,int(y_pred))
        return render_template("index.html", name=name)
@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({
                'response': 'Please provide a message.',
                'status': 'error'
            })
        
        print(f"Received message: {user_message}")  # Debug log
        
        # Create a context about phishing for the AI
        context = """You are a helpful AI assistant specializing in cybersecurity and phishing detection. 
        Provide clear, accurate information about phishing threats, prevention, and best practices. 
        Keep responses focused on cybersecurity and phishing-related topics."""
        
        # Combine context and user message
        prompt = f"{context}\n\nUser: {user_message}\nAssistant:"
        print(f"Sending prompt to Gemini...")  # Debug log
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        print(f"Received response from Gemini")  # Debug log
        
        if not response or not hasattr(response, 'text'):
            return jsonify({
                'response': 'The AI model returned an empty response. Please try again.',
                'status': 'error'
            })
        
        return jsonify({
            'response': response.text,
            'status': 'success'
        })
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in ask_assistant: {str(e)}\nTraceback: {error_traceback}")  # Debug log
        return jsonify({
            'response': f"An error occurred: {str(e)}. Please try again.",
            'status': 'error'
        })
@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')
if __name__ == "__main__":
    app.run(debug=True)
