import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('titanicModel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    feature = [x for x in request.form.values()]
    # feature[age,sib,parch,class,gen,embark]
    embarked = {'S': 0, 'C': 1, 'Q': 2}
    gen = {'male': 0, 'female': 1}
    if feature[4] == 'male':
        feature[4] = 0
    else:
        feature[4] = 1

    if feature[5] == 'S':
        feature[5] = 0
    elif feature[5] == 'C':
        feature[5] = 1
    else:
        feature[5] = 2
    int_feature = [int(i) for i in feature]
    final_feature = [np.array(int_feature)]
    prediction = model.predict(final_feature)

    if prediction == 0:
        output = 'not survive'
    else:
        output = 'survive'

    return render_template('index.html', prediction_text='You will {} the titanic crash.'.format(output))


if __name__ == "__main__":
    app.run(debug=True)