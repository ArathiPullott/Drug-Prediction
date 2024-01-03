
# importing the necessary dependencies
from flask import Flask, render_template, request
import pickle
import numpy as np


application = Flask(__name__) # initializing a flask app
# app=application
@application.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@application.route('/predict',methods=['POST']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user

            gender=request.form['gender']
            bp = request.form['bp']
            cholesterol =request.form['cholesterol']
            age = request.form['age']
            natok = request.form['natok']
            inPut = np.array([[gender, bp, cholesterol, age, natok]])
            print(inPut)
            encoder=pickle.load(open('finalEncoder.pkl', 'rb'))
            model=pickle.load(open('finalModel.pkl', 'rb'))
            # predictions using the loaded model file
            encoded_data = encoder.transform(inPut)
            # print(encoded_data)
            prediction=model.predict(encoded_data)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html', prediction=prediction[0])
        except Exception as e:
            print('An Exception Occurred: ', str(e))
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	application.run(debug=True,port=9000) # running the app