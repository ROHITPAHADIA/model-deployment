from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        email_text = request.form.get("email-content")
        tokenized_email = tokenizer.transform([email_text])  # Wrap email_text in a list
        predictions = model.predict(tokenized_email)[0]  # Get the first value of predictions
        predictions = 1 if predictions == 1 else -1
        return render_template("index.html", predictions=predictions, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
