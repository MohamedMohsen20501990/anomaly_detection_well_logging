import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import pandas as pd
from src.logging_config import logger

@dataclass
class PreprocessConfig:
    """Configuration class containing paths for preprocessed data artifacts."""
    
    transformed_data_path : str = os.path.join("artifacts", "transformed_data.parquet")
    
class Preprocessor:
    """Handles missing-value removal and saving of preprocessed datasets."""
    def __init__(self):
        self.preprocess_config = PreprocessConfig()
        
    def apply_and_save(self, data: pd.DataFrame):
        """Remove rows containing missing values and save the resulting dataset.

        The method drops all observations containing at least one missing value,
        saves the remaining complete observations as a Parquet file, and returns
        the processed DataFrame.

        Args:
            data: Input pandas DataFrame to preprocess.

        Returns:
            pd.DataFrame: Processed DataFrame containing only complete observations,
            with the original column names and row indices preserved.
        """
        
        os.makedirs(os.path.dirname(self.preprocess_config.transformed_data_path), exist_ok=True)
        
        transformed_df = data.dropna().copy().reset_index(drop=True)
        
        transformed_df.to_parquet(self.preprocess_config.transformed_data_path)
        
        logger.info(
                        f"Transformed data successfully saved\n"
                        f"Total transformed samples: {len(transformed_df)}"
                    )
        
        return transformed_df    

    