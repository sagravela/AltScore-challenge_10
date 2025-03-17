import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s-%(name)s-%(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parents[0]
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
EXTERNAL_DATA_DIR = BASE_DIR / 'data' / 'external'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DATA_DIR = BASE_DIR / 'model'
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# File names
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
MOBILITY_FILE = 'mobility_data.parquet'
AGG_MOBILITY_FILE = 'agg_mobility.csv'
FE_FILE = 'fe_data.csv'
FULL_DATASET = 'full_dataset.csv'
STUDY_FILE = 'study.log'
PARAMS_FILE = 'params.json'
MODEL_FILE = 'model.joblib'
SUBMISSION_FILE = 'submission.csv'

# Hex resolution
RESOLUTION = 8

# External data source configuration
# Nominatim
NOMINATIM_PATH = EXTERNAL_DATA_DIR / 'nominatim'
NOMINATIM_PATH.mkdir(parents=True, exist_ok=True)
NOMINATIM_FILE = 'addresses.csv'
# Numbeo
NUMBEO_PATH = EXTERNAL_DATA_DIR / 'numbeo'
NUMBEO_PATH.mkdir(parents=True, exist_ok=True)
NUMBEO_PRICES_FILE = 'prices.csv'
NUMBEO_QUALITY_FILE = 'quality_of_life.csv'
NUMBEO_CITIES = [
    "Cuenca",
    "Guayaquil",
    "Quito"
]
# Censo 2022
CENSUS_PATH = EXTERNAL_DATA_DIR / 'censo-EC-2022'
CENSUS_FILE = 'census-2022.csv'
CENSUS_POP_FILE = '01_2022_CPV_Estructura_poblacional.xlsx'
CENSUS_POOR_FILE = '2022_CPV_Pobreza-por-NBI.xlsx'
CENSUS_HOUSE_APPLIANCES_FILE = '01_2022_CPV_CONDICIONES_DE_VIDA.xlsx'
CENSUS_HOUSE_TIC_FILE = '03_2022_CPV_TIC.xlsx'
