import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.schemas import InputData
from src.utils import load_obj
from src.logging_config import logger

class InferencePipeline:
    """
    Handles loading the trained model, preparing input data,
    and generating anomaly detection predictions.
    """
    FEATURE_COLUMNS = (
    "NPHI",
    "RHOB",
    "GR",
    "CALI",
    "PEF",
    "DTC",
    "RDEP",
    "DEPTH_MD"
    )
    LABELS = {
        1: "normal",
        -1: "anomaly"
    }
    
    def __init__(self, model_path: str | Path):
        self.model = load_obj(model_path)
        
    @staticmethod    
    def _prepare_input(input_features: InputData):
        """
        Convert validated input features into a DataFrame
        with the expected feature ordering.

        Args:
            input_features: Validated input features.

        Returns:
            pd.DataFrame: Prepared input DataFrame.
        """
        features = input_features.model_dump()
        
        df = pd.DataFrame(features, columns=InferencePipeline.FEATURE_COLUMNS)    
        return df
        
        
    def predict(self, input_data:pd.DataFrame | dict):
        """
        Validate input data and generate an anomaly detection prediction.

        Args:
            input_data (dict): Dictionary containing the required input features.

        Returns:
            dict: Dictionary containing the model prediction.
                  1 represents a normal observation.
                 -1 represents an anomalous observation.

        Raises:
            Exception: If an error occurs during input validation or prediction.
        """
        try:
            
            if isinstance(input_data, dict):
                input_features = InputData(**input_data)
                df = self._prepare_input(input_features)

            elif isinstance(input_data, pd.DataFrame):
                df = input_data.loc[:, self.FEATURE_COLUMNS]
                

            else:
                raise TypeError(
                    "Input must be a pandas DataFrame or dictionary."
                )
            
            predictions = self.model.predict(df)
            
            results = [self.LABELS[int(prediction)]for prediction in predictions]
            
            
            
            return {"Predictions":results}
        except Exception:
            logger.exception("Error while making prediction")
            raise
        
if __name__ == "__main__":
    inference_pipeline = InferencePipeline(model_path="./artifacts/model.pkl")
    prediction = inference_pipeline.predict({
    "NPHI": [0.45, 0.2, 0.4],
    "RHOB": [2.35, 2.40, 2.30],
    "GR": [75.2, 82.5, 68.4],
    "CALI": [8.5, 8.7, 8.3],
    "PEF": [3.1, 2.9, 3.2],
    "DTC": [95.0, 102.5, 91.0],
    "RDEP": [12.5, 15.2, 10.8],
    "DEPTH_MD": [1500.0, 1501.0, 1502.0]
    })
   
    # df = pd.read_csv("./artifacts/train.csv")
    # prediction = inference_pipeline.predict(df)

    # prediction = pd.Series(prediction).map({1:"normal", -1:"anomaly"}).iloc[0]
    
    print(prediction)  