# app_flask.py
from flask import Flask, render_template, request, jsonify
from model_infer import predict_sentiment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("post_text", "")
        result = predict_sentiment(text)
    return render_template("index.html", result=result, text=text)


@app.route("/api/predict", methods=["POST"])  # JSON API
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = predict_sentiment(text)
    return jsonify(result)


if __name__ == "__main__":
    # For production use a WSGI server like waitress or gunicorn
    # waitress-serve --call app_flask:app
    app.run(debug=True)