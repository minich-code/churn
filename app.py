from flask import Flask, request, render_template
import os 

from src.churn.pipelines.pip_07_prediction_pipeline import CustomData, PredictionPipeline
from src.churn.exception import FileOperationError
from src.churn import logging

# Create a flask app
app = Flask(__name__)

# Route for homepage 
@app.route('/')
def home():
    return render_template('home.html')

# Route for prediction

@app.route('/predict', methods=['GET', 'POST'])
def predict_data_point():
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        try:
            # Initialize an empty dictionary to store the data
            form_data = {}

            # Iterate over the form fields and populate the dictionary
            for field in [  'age'  
                            'gender',
                            'region_category',
                            'membership_category',
                            'joined_through_referral',
                            'preferred_offer_types',
                            'internet_option',
                            'Recency',
                            'avg_time_spent',
                            'Monetary', 
                            'Frequency',
                            'points_in_wallet', 
                            'used_special_discount',
                            'offer_application_preference',
                            'past_complaint',
                            'complaint_status',
                            'feedback',
                            'churn_risk_score',
                            'medium_of_operation',
                            'tenure_months',
                            'visit_hour',
                            'Login-Spend Ratio',
                            'Login-Transaction Ratio', 
            ]:
                form_data[field] = request.form.get(field)


            # Create an instance of the CustomData class
            custom_data = CustomData(**form_data)

            # Convert the form data dictionary to a dataframe 
            pred_df = custom_data.get_data_as_dataframe()

            # Print the dataframe for debugging 
            print(pred_df)

            # Log message 
            logging.info(f"Form data before prediction: {form_data}")

            # Initialize the prediction pipeline 
            prediction_pipeline = PredictionPipeline()

             # Get the prediction 
            prediction = prediction_pipeline.make_predictions(pred_df)

            # Return results 
            return render_template('prediction.html', prediction=prediction[0])
        
        except Exception as e:
            logging.exception(e)
            #raise FileOperationError(e)
            return render_template('predict.html', error_message="Please enter valid numbers for all fields.")
        

# Run the flask app 

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=8080, debug=True)

    