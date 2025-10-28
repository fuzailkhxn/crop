from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = request.form['Nitrogen']
        P = request.form['Phosporus']
        K = request.form['Potassium']
        temp = request.form['Temperature']
        humidity = request.form['Humidity']
        ph = request.form['pH']
        rainfall = request.form['Rainfall']

        # Check for missing input
        if not all([N, P, K, temp, humidity, ph, rainfall]):
            raise ValueError("All fields must be filled.")

        # Convert to float (will error if not valid number)
        feature_list = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
        crop_image_dict = {
            "Rice": "rice.png",
            "Maize": "maize.png",
            "Jute": "jute.png",
            "Cotton": "cotton.png",
            "Coconut": "coconut.png",
            "Papaya": "papaya.png",
            "Orange": "orange.png",
            "Apple": "apple.png",
            "Muskmelon": "muskmelon.png",
            "Watermelon": "watermelon.png",
            "Grapes": "grapes.png",
            "Mango": "mango.png",
            "Banana": "banana.png",
            "Pomegranate": "pomegranate.png",
            "Lentil": "lentil.png",
            "Blackgram": "blackgram.png",
            "Mungbean": "mungbean.png",
            "Mothbeans": "mothbeans.png",
            "Pigeonpeas": "pigeonpeas.png",
            "Kidneybeans": "kidneybeans.png",
            "Chickpea": "chickpea.png",
            "Coffee": "coffee.png"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
            crop_image = crop_image_dict.get(crop, "default.png")
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            crop_image = "default.png"

        return render_template('index.html', result=result, crop_image=crop_image)

    except Exception as e:
        error_message = f"Input error: {str(e)}. Please ensure all fields are filled with valid numbers in the allowed range."
        crop_image = "default.png"
        return render_template('index.html', result=error_message, crop_image=crop_image)



if __name__ == "__main__":
    app.run(debug=True)