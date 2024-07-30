from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

app=Flask(__name__)

# Define a function to preprocess the input data
def preprocess_input(input_str):
    input_list = input_str.split(',')
    input_array = np.array(input_list, dtype=np.float32)
    return input_array.reshape(1, -1)

#flask app
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    input_str=request.form['features']
    input_array = preprocess_input(input_str)
    prediction = model.predict(input_array)[0]


    output_message=f" the sales prediction is: {prediction:.2f}"

    return render_template('index.html', message=output_message)


if __name__ == '__main__':
    app.run(debug=True)