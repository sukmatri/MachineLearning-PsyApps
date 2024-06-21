from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

LABEL = ["Regardless of age, investing in mental health is the best gift you can give yourself.",
         "No matter your gender, taking care of your mental health is a brave step towards true happiness.",
         "Dont hesitate to talk to your family about your mental health. They are an invaluable pillar of support.",
         "Deciding to seek treatment is the first and most crucial step in your journey towards better mental health.",
         "Find a balance between work and mental health. Work becomes lighter when your mind is at peace.",
         "You have the right to choose the best care options for your mental health. Dont be afraid to seek what you need.",
         "Take advantage of wellness programs at your workplace. Its a small investment that can bring significant changes to your life.",
         "Seeking help does not show weakness, but your strength to face and overcome mental challenges.",
         "Ignoring mental health can have long-term consequences. Its better to prevent than to cure.",
         "Good mental health supports your physical health. Dont neglect one for the other; both are equally important."]

# Load the model
model = load_model('model/model_klasifikasi.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/klasifikasi', methods=['POST'])
def predict_klasifikasi():
    # Mengambil data dari form
    age = request.form.get('age', 0.0, type=float)
    gender = request.form.get('gender', 0.0, type=float)
    family = request.form.get('family', 0.0, type=float)
    treatment = request.form.get('treatment', 0.0, type=float)
    work = request.form.get('work', 0.0, type=float)
    care = request.form.get('care', 0.0, type=float)
    wellness = request.form.get('wellness', 0.0, type=float)
    seek = request.form.get('seek', 0.0, type=float)
    mental = request.form.get('mental', 0.0, type=float)
    phys = request.form.get('phys', 0.0, type=float)
    
    new_data = np.array([[age, gender, family, treatment, work, care, wellness, seek, mental, phys]])
    
    # Perform prediction
    prediction = model.predict(new_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = LABEL[predicted_class]
    print(result)
    
    return render_template('result.html', 
                           age=age,
                           gender=gender,
                           family=family,
                           treatment=treatment,
                           work=work,
                           care=care,
                           wellness=wellness,
                           seek=seek,
                           mental=mental,
                           phys=phys,
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)
