from flask import Flask, request, jsonify
import pandas as pd
import json
from flask_mail import Mail, Message

from src.churn.pipelines.pip_07_prediction_pipeline import CustomData, PredictionPipeline
from src.churn.exception import FileOperationError
from src.churn import logging

# Create a flask app
app = Flask(__name__)

# Mailtrap configuration
app.config['MAIL_SERVER'] = 'bulk.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'api'
app.config['MAIL_PASSWORD'] = '6d0d5e3485f30cbe048871e1b9c7c2c9'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict_data_point():
    if request.method == 'POST':
        try:
            # Debug statement to check if the request has files
            if 'csv_file' not in request.files:
                logging.error("No csv_file key in request.files")
                return jsonify({"error": "No file part in the request"}), 400

            csv_file = request.files['csv_file']

            # Check if a file was uploaded
            if csv_file.filename == '':
                logging.error("No selected file")
                return jsonify({"error": "No file selected"}), 400

            # Read the CSV data into a Pandas DataFrame
            df = pd.read_csv(csv_file)

            # Select the first row for prediction
            first_row = df.iloc[1]

            # Convert the first row to a dictionary
            first_row_data = first_row.to_dict()

            # Create CustomData instance from the first row data
            custom_data = CustomData(**first_row_data)

            # Get data as a DataFrame (or use the existing df directly if needed)
            pred_df = custom_data.get_data_as_dataframe()

            # Log the data for debugging
            logging.info(f"Data received for prediction: {first_row_data}")

            # Initialize prediction pipeline
            prediction_pipeline = PredictionPipeline()

            # Get the prediction 
            prediction = prediction_pipeline.make_predictions(pred_df)

            # Send email based on prediction
            send_email(prediction[0])

            # Return prediction as JSON
            return jsonify({"prediction": prediction[0]})
        except Exception as e:
            logging.exception(e)
            return jsonify({"error": "An error occurred during prediction."}), 500  # Return error code 500
    else:
        return '''
            <!DOCTYPE html>
            <html>
            <head>
              <title>Churn Prediction</title>
            </head>
            <body>
              <h1>Churn Prediction</h1>
              <form method="POST" action="/predict" enctype="multipart/form-data"> 
                <input type="file" name="csv_file" accept=".csv">
                <button type="submit">Predict</button>
              </form>
            </body>
            </html>
        '''

def send_email(prediction):
    # Choose email content based on prediction category
    if prediction == 2:
        subject = "Your Subscription is Safe!"
        text = "We are happy to inform you that your subscription is in good standing."
    elif prediction == 1:
        subject = "Subscription Renewal Alert"
        text = "Your subscription is nearing renewal. Please take the necessary actions."
    else:
        subject = "Subscription Cancellation Warning"
        text = "We noticed some issues with your subscription. Please contact support."

    # Create a message object
    msg = Message(subject, sender='mailtrap@demomailtrap.com', recipients=['minichworks@gmail.com'])
    msg.body = text

    # Send email using Mailtrap
    with app.app_context():
        mail.send(msg)

# Run the flask app 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
