import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.logging_config import logger
from src.utils import save_obj

@dataclass
class TrainerConfig:
    """Configuration class for model training."""
    model_path: str = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    """Handles training and saving the Isolation Forest model."""
    def __init__(self):
        self.trainer_config = TrainerConfig()
        
    def train_model(self, data_path: str | Path):
        """
        Train an Isolation Forest model using transformed data.

        Args:
            data_path (str | Path): Path to the transformed Parquet dataset.

        Returns:
            IsolationForest: The trained Isolation Forest model.

        Raises:
            Exception: If an error occurs during the training process.
        """
        try:
            df = pd.read_parquet(data_path)
            
            if_model = IsolationForest(contamination=0.1, random_state=42)
            if_model.fit(df)
            save_obj(if_model, self.trainer_config.model_path)       
            logger.info(f"Model has been trained and saved successfully to {self.trainer_config.model_path}")
            return if_model
        except Exception:
            logger.exception("Error while training process")
            raise