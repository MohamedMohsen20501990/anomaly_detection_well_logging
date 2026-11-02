import sys
import os
from typing import Literal, TypeAlias
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.logging_config import logger
from dataclasses import dataclass

from src.preprocess import Preprocessor
from src.train import ModelTrainer

@dataclass
class DataConfig:
    """Configuration class containing paths for training and testing datasets."""
    
    train_data_path: str | Path = os.path.join("artifacts", "train.csv")

    
FEATURES: TypeAlias = Literal["NPHI", "RHOB", "GR", "CALI", "PEF", "DTC", "RDEP", "DEPTH_MD"]
default_features: list[FEATURES] = ["NPHI", "RHOB", "GR", "CALI", "PEF", "DTC", "RDEP", "DEPTH_MD"]
    
class DataPipeline:
    """Handles data loading, preprocessing, splitting, and saving."""

    def __init__(self):
        self.data_config = DataConfig()
        
    
    @staticmethod
    def _wrangle_data(df: pd.DataFrame, features: list[FEATURES]| None = None):
        """Select specified features from a DataFrame or return a copy of the full dataset.

        Args:
            df: Input pandas DataFrame.
            features: Optional list of feature names to select.

        Returns:
            A DataFrame containing the selected features or a copy of the input DataFrame.
        """
        
        if not features:
            features = default_features
            
        wrangled_data = df[features]
        return wrangled_data
            
            
        
    def read_csv_data(self, path: str | Path, input_features:list[FEATURES]| None = None):
        """Read a CSV file and optionally select specific input features.

        Args:
            path: Path to the CSV file.
            input_features: Optional list of features to retain from the dataset.

        Returns:
            A DataFrame containing the requested features or the complete dataset.
        """
        
        df = pd.read_csv(path)
        logger.info("Reading data from %s", path)
        return self._wrangle_data(df, input_features)  
    
    
    def save_data(self, data: pd.DataFrame):
        """Save the entire dataset as a CSV file without splitting it.

        Args:
            data: Input pandas DataFrame to save.

        Returns:
            The original DataFrame.

        Raises:
            Exception: If an error occurs while saving the dataset.
        """
        try:
            os.makedirs(os.path.dirname(self.data_config.train_data_path),exist_ok=True)

            data.to_csv(self.data_config.train_data_path,index=False,header=True)

            logger.info(
                f"Data successfully saved\n"
                f"Total samples: {len(data)}"
            )

            return data

        except Exception:
            logger.exception("Error saving data")
            raise

        
if __name__ == "__main__":
    data_pipeline = DataPipeline()
    df = data_pipeline.read_csv_data(path="./data/Xeek_Well_15-9-15.csv")
    train_set = data_pipeline.save_data(df)   
    
    preprocessor = Preprocessor()
    transformed_data = preprocessor.apply_and_save(data = train_set)
    
    trainer = ModelTrainer()
    trainer.train_model(data_path=preprocessor.preprocess_config.transformed_data_path)
        
        
    

