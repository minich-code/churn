from src.churn.config.configuration import ConfigurationManager
from src.churn.components.c_02_data_validation import DataValidation
from src.churn import logging
import pandas as pd 

PIPELINE_NAME = "DATA VALIDATION PIPELINE"

class DataValidationPipeline:
    def __init__(self):
        pass

    def run(self):
        # Initialize ConfigurationManager to get configuration settings
        config = ConfigurationManager()
        # Get data validation configuration
        data_validation_config = config.get_data_validation_config()
        # Create DataValidation object with the obtained configuration
        data_validation = DataValidation(config=data_validation_config)

        # Fetch data from CSV (or alternatively directly from MongoDB)
        data = pd.read_csv(data_validation_config.data_dir)
        
        # Perform data validation for all columns
        column_validation_status = data_validation.validate_all_columns(data)
        
        # Perform data type validation for all columns
        type_validation_status = data_validation.validate_data_types(data)

        # Perform missing values validation
        missing_values_status = data_validation.validate_missing_values(data)

        overall_validation_status = (column_validation_status and type_validation_status and 
                                     missing_values_status)

        if overall_validation_status:
            logging.info("Data Validation Completed Successfully!")
        else:
            logging.info("Data Validation Failed. Check the status file for more details.")

if __name__ == "__main__":
    try:
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()
    except Exception as e:
        logging.error(f"Data validation pipeline failed: {e}")
        raise