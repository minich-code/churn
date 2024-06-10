from dataclasses import dataclass 
from pathlib import Path
import pymongo
from pymongo import MongoClient 
import pandas as pd
import os 


# Data Ingestion entity 
@dataclass
class DataIngestionConfig:
    root_dir: Path
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int = 10000  # Batch size for batch processing


# Data validation entity
@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict
    critical_columns: list  # List of critical columns for missing value checks


# Data transformation entity
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: list
    categorical_cols: list

