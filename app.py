from flask import Flask, render_template, request, redirect, url_for
from joblib import load
from model.get_tweets import get_related_tweets
import pickle

# model = load("model/text_classification.joblib")

# Use pickle to load in the pre-trained model.
with open(f'model/project_virago_xgboost.pkl', 'rb') as f:
    pipeline = pickle.load(f)

def requestResults(name):
    tweets = get_related_tweets(name)
    tweets['prediction'] = pipeline.predict(tweets['tweet_text'])
    data = str(tweets.prediction.value_counts()) + '\n\n'
    return data + str(tweets)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/get', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))


@app.route('/success/<name>')
def success(name):
    return "<xmp>"+str(requestResults(name)) + "<xmp>"


if __name__ == '__main__' :
    app.run(debug=True)