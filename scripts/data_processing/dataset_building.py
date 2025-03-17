import polars as pl
from thefuzz import process

from scripts import logging
from scripts.external.census_2022 import spell_column
from scripts import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_FILE, TEST_FILE, 
    AGG_MOBILITY_FILE, FE_FILE, FULL_DATASET, NOMINATIM_PATH, NOMINATIM_FILE, 
    CENSUS_PATH, CENSUS_FILE, NUMBEO_PATH, NUMBEO_PRICES_FILE, NUMBEO_QUALITY_FILE,
)

def load_data() -> tuple:
    """Load all the data from `data` folder."""
    logging.info(f"Loading data from {RAW_DATA_DIR}.")
    raw_train = pl.read_csv(RAW_DATA_DIR / TRAIN_FILE)
    raw_test = pl.read_csv(RAW_DATA_DIR / TEST_FILE)

    logging.info(f"Loading data from {PROCESSED_DATA_DIR}.")
    agg_mobility_df = pl.read_csv(PROCESSED_DATA_DIR / AGG_MOBILITY_FILE)
    fe_df = pl.read_csv(PROCESSED_DATA_DIR / FE_FILE)
    
    logging.info(f"Loading data from {NOMINATIM_PATH}.")
    nominatim_df = pl.read_csv(NOMINATIM_PATH / NOMINATIM_FILE, infer_schema_length=None)

    logging.info(f"Loading data from {CENSUS_PATH}.")
    census_df = pl.read_csv(CENSUS_PATH / CENSUS_FILE)

    logging.info(f"Loading data from {NUMBEO_PATH}.")
    prices_df = pl.read_csv(NUMBEO_PATH / NUMBEO_PRICES_FILE)
    quality_df = pl.read_csv(NUMBEO_PATH / NUMBEO_QUALITY_FILE)

    return raw_train, raw_test, agg_mobility_df, census_df, nominatim_df, fe_df, prices_df, quality_df

def prepare_nominatim_data(nominatim_df: pl.DataFrame, reference_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the Nominatim data for merging. This does the following:
    - Map ISO codes to provinces.
    - Coalesce columns in `province`, `county`, `district` and `neighbourhood`.
    - Replace some edge cases and correct spelling.
    """

    def correct_spelling(value, reference_list):
        """Function to correct spelling using fuzzy matching"""
        match, score = process.extractOne(value, reference_list)
        return match if score > 87 else value  # I've tuned the threshold here
    
    # Source https://es.wikipedia.org/wiki/ISO_3166-2:EC
    code_to_province = {
        "EC-A": "Azuay",
        "EC-B": "Bolívar",
        "EC-F": "Cañar",
        "EC-C": "Carchi",
        "EC-H": "Chimborazo",
        "EC-X": "Cotopaxi",
        "EC-O": "El Oro",
        "EC-E": "Esmeraldas",
        "EC-W": "Galápagos",
        "EC-G": "Guayas",
        "EC-I": "Imbabura",
        "EC-L": "Loja",
        "EC-R": "Los Ríos",
        "EC-M": "Manabí",
        "EC-S": "Morona Santiago",
        "EC-N": "Napo",
        "EC-D": "Orellana",
        "EC-Y": "Pastaza",
        "EC-P": "Pichincha",
        "EC-SE": "Santa Elena",
        "EC-SD": "Santo Domingo de los Tsáchilas",
        "EC-U": "Sucumbíos",
        "EC-T": "Tungurahua",
        "EC-Z": "Zamora Chinchipe"
    }

    return (
        nominatim_df
        .select(
            pl.col('hex_id'),
            pl.col('ISO3166-2-lvl4').replace_strict(code_to_province).alias('province'),
            # Coalesce columns
            pl.coalesce(pl.col(['county', 'city', 'town'])).alias('county'),
            pl.coalesce(pl.col(['city', 'town', 'city_district', 'municipality', 'village'])).alias("district"),
            pl.coalesce(pl.col(['locality', 'suburb', 'neighbourhood', 'hamlet', 'isolated_dwelling', 'quarter'])).alias("neighbourhood"),
        )
        .with_columns([
            pl.col('province').str.replace('Santo Domingo de los Tsáchilas', 'Santo Domingo De Los Tsáchilas'),
            ## Disclaimer: The following replacements could be wrong. I'm not familiar with the geography of Ecuador.
            (
                pl.col('county')
                .str.replace('Sangolqui', 'Rumiñahui')
                .str.replace('Machachi', 'Mejía')
            ),
            (
                pl.col('district')
                .str.replace('^Parroquia ', '')  # Remove prefix
                .str.replace('Ibarra', 'San Miguel De Ibarra')
                .str.replace('Durán', 'Eloy Alfaro')
                .str.replace('Santo Domingo', 'Santo Domingo De Los Colorados')
            )
        ])
        .with_columns(
            # Spelling correction based on census data
            [spell_column(reference_df, col, 87) for col in ['province', 'county', 'district']]
        )
    )

def prepare_census_data(census_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the census data for merging. Transform some census features for their ratios."""
    def calculate_ratio(col1: str, col2: str) -> pl.Expr:
        return (pl.col(col1) / (pl.col(col1) + pl.col(col2)))
    
    features_y = census_df.select(pl.selectors.ends_with('_y')).columns
    features_n = census_df.select(pl.selectors.ends_with('_n')).columns
    return (
        census_df
        .with_columns(
            # Calculate ratios
            [calculate_ratio(x, y).alias(x) for x, y in zip(features_y, features_n)]
        )
        .drop(pl.selectors.ends_with('_n'))
    )

def prepare_prices_data(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the prices data for merging. This does the following:
    - Remove hyphens.
    - Cast to float.
    - Fill missing values with mean.
    - Aggregate by city and take mean.
    """
    return (
        df
        .with_columns(pl.col(pl.String).replace('-', None))
        .with_columns(pl.col(df.drop('city').columns).cast(pl.Float64()))
        .with_columns(pl.all().fill_null(strategy="mean"))
        .group_by('city')
        .mean()
    )

def merge_data(raw_train, raw_test, agg_mobility_df, census_df, nominatim_df, fe_df, prices_df, quality_df) -> pl.DataFrame:
    """Merge all datasets while ensuring consistency of the data."""
    
    # Merge Nominatim and census data to ensure not missing districts
    nominatim_census_df = (
        nominatim_df
        .join(census_df, on=['province', 'county', 'district'], how='left')
        # Sometimes the value of `county` and `district` are the same and Nominatim only gives you one of them
        # Replace 'district' with 'county' where 'population' is null
        .with_columns(
            pl.when(pl.col('pop_density_km2').is_null())
            .then(pl.col('county'))
            .otherwise(pl.col('district'))
            .alias('district')
        )
        # Select specific columns and rejoin with census data
        .select(['hex_id', 'province', 'county', 'district', 'neighbourhood'])
        .join(census_df, on=['province', 'county', 'district'], how='left')
    )

    # Flag each dataset and cast cost_of_living to float
    train = raw_train.with_columns(pl.lit(True).alias('train'))
    test = raw_test.with_columns(
        pl.col('cost_of_living').cast(pl.Float64()),
        pl.lit(False).alias('train')
    )
    
    logging.info("Merging data...")
    # Merge train and test datasets
    full_data = (
        pl.concat([train, test])
        .join(nominatim_census_df, on='hex_id', how='left')
        .join(agg_mobility_df, on='hex_id', how='left')
        .join(fe_df, on='hex_id', how='left')
        .join(quality_df, left_on='closest_city', right_on='city', how='left')
        .join(prices_df, left_on='closest_city', right_on='city', how='left')
    )

    return full_data


if __name__ == '__main__':
    raw_train, raw_test, agg_mobility_df, census_df, nominatim_df, fe_df, prices_df, quality_df= load_data()

    logging.info("Preparing data...")
    proc_census_df = prepare_census_data(census_df)
    reference_df = proc_census_df.select(['province', 'county', 'district'])
    proc_nominatim_df = prepare_nominatim_data(nominatim_df, reference_df)
    proc_prices_df = prepare_prices_data(prices_df)
    full_data = merge_data(raw_train, raw_test, agg_mobility_df, proc_census_df, proc_nominatim_df, fe_df, proc_prices_df, quality_df)

    # Save processed data
    logging.info(f"Saving processed data to {PROCESSED_DATA_DIR / FULL_DATASET}.")
    full_data.write_csv(PROCESSED_DATA_DIR / FULL_DATASET)
