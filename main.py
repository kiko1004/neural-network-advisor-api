import yfinance as yf
from flask import Flask
from flask import jsonify, make_response

from NNT import Predictor

app = Flask(__name__)

@app.route("/")
def home():
    return '<h1 style="text-align:center;">Neural Network Trading Advisor API</h1>' \
           '<h2 style="text-align:center;">By Kiril Spiridonov</h2>' \
           '<h5 style="text-align:center;">Endpoints: /ticker/*ticker name here* </h5>'


@app.route("/ticker/<ticker>")
def analyze(ticker):
    _ticker = yf.Ticker(ticker)
    df = _ticker.history(period='1y').reset_index()
    predictor = Predictor(df = df)
    binary_prediction, binary_confidence, interval_prediction, interval_confidence = predictor.predict_next_week()
    info = _ticker.info['shortName']
    suggestion = "Should probably wait for a better setup or change the ticker."
    if (binary_prediction == 1) & (interval_prediction in ['1% - 2%', '2% - 3%', '3%+']):
        suggestion = "Probably a good buy. Look at the confidence of our advisors."


    response = make_response(
        jsonify(
            {"Asset_Name": info,
             "Neural_Network_1": str(binary_prediction),
             "Neural_Network_1_confidence": str(binary_confidence),
             "Neural_Network_2": str(interval_prediction),
             "Neural_Network_2_confidence": str(interval_confidence),
             "Strategy_Suggestion": suggestion}
        ),
        200,
    )
    response.headers["Content-Type"] = "application/json"
    return response

if __name__ == "__main__":
    app.run(debug=True)