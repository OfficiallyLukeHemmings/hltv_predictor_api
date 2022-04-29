from flask import Flask, request, Response, jsonify
import pandas as pd
import model # model.py import

# Initialising the ML models, and training the models ready for predictions.
# pos meaning positively favouring neutral scores (already converted in csv)
posClf = model.Predictor()
posClf.train()

# neg meaning negatively favouring neutral scores (already converted in csv)
negClf = model.Predictor(False)
negClf.train()

# Flask API
app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    try:
        game_json = request.json        
        df = pd.DataFrame([game_json])
        
        prediction = jsonify({
            'optimistic-prediction': posClf.pred(df),
            'pessimistic-prediction': negClf.pred(df)
        })
        # status code: 200 OK - The resource describing the result of the 
        # action is transmitted in the message body.
        return prediction, 200
    
    except Exception as e:
        print('Error encountered:', e)
        # status code: 400 BAD REQUEST - The server cannot or will not process
        # the request due to something that is perceived to be a client error
        # e.g. malformed request syntax
        return {}, 400

if __name__ == '__main__':
    app.run()