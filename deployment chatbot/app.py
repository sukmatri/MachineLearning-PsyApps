from flask import Flask, request, render_template
import joblib
import numpy as np
from tensorflow import keras


app = Flask(__name__)

LABEL = ["Hello there. Tell me how are you feeling today?", "Hi there. What brings you here today?", "Hi there. How are you feeling today?", "Great to see you. How do you feel currently?", "Hello there. Glad to see you're back. What's going on in your world right now?",
         "Just as there are different types of medications for physical illness, different treatment options are available for individuals with mental illness. Treatment works differently for different people. It is important to find what works best for you or your child."]

# Load the model
model = joblib.load('./chatbotmodel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/klasifikasi', methods=['POST'])
def predict_klasifikasi():
    # Mengambil data dari form
    age = request.form.get('patterns', 0.0, type=float)

    new_data = np.array([[patterns]])
    
    # Perform prediction
    prediction = model.predict(new_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = LABEL[predicted_class]
    print(result)
    
    return render_template('result.html', 
                           patterns=patterns,
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)