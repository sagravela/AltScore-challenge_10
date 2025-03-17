import polars as pl
from thefuzz import process

from scripts import logging
from scripts import (
    CENSUS_PATH, CENSUS_FILE, CENSUS_POP_FILE, CENSUS_POOR_FILE, 
    CENSUS_HOUSE_APPLIANCES_FILE, CENSUS_HOUSE_TIC_FILE
)

def load_data():
    """Load census data from `CENSUS_PATH`."""
    logging.info(f"Loading data from {CENSUS_PATH}.")
    pop_df = pl.read_excel(CENSUS_PATH / CENSUS_POP_FILE, sheet_name='4.2')
    poor_df = pl.read_excel(CENSUS_PATH / CENSUS_POOR_FILE, sheet_name='1.2')
    house_appliances_df = pl.read_excel(CENSUS_PATH / CENSUS_HOUSE_APPLIANCES_FILE, sheet_name='7.1')
    house_tic_df = pl.read_excel(CENSUS_PATH / CENSUS_HOUSE_TIC_FILE, sheet_name='3.2')
    return pop_df, poor_df, house_appliances_df, house_tic_df

def clean_data(df: pl.DataFrame, cols_drop_ind: list[int], new_cols: list[str]) -> pl.DataFrame:
    """Function to clean census data.

    Parameters:
        df (pl.DataFrame): The dataframe to clean.
        cols_drop_ind (list[int]): A list of indices of columns to drop.
        new_cols (list[str]): A list of new column names.
    """
    return (
        df
        .drop([df.columns[i] for i in cols_drop_ind])  # Drop the first column
        .rename({old: new for old, new in zip([c for c in df.columns if df.columns.index(c) not in cols_drop_ind], new_cols)})
        # Replace any text in `county` with 'Quito' word as 'Quito'
        .with_columns(
            pl.when(pl.col('county').str.contains('Quito'))
            .then(pl.lit('Quito'))
            .otherwise(pl.col('county'))
            .alias('county')
        )
    )

def spell_column(reference_df: pl.DataFrame, col: str, score: int) -> pl.Expr:
    """Polars expresion to correct spelling using fuzzy matching.

    Parameters:
        reference_df (pl.DataFrame): The reference dataframe which contains the vocabulary.
        col (str): The column to correct spelling.
        score (int): The score threshold for fuzzy matching.
    """
    def correct_spelling(value, reference_list, threshold = 80):
        """Helper function to correct spelling using fuzzy matching"""
        match, score = process.extractOne(value, reference_list)
        return match if score > threshold else value  # I've tuned the threshold
    return (
        # Correct spelling
        pl.col(col)
        .map_elements(lambda x: correct_spelling(x, reference_df[col].to_list(), score), return_dtype=pl.String())
    )

def prep_pop_data(pop_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the census population data.

    Parameters:
        pop_df (pl.DataFrame): The dataframe to prepare.
    """
    new_cols = ['province', 'county', 'district', 'pop_density_km2']
    cols_drop_ind = [0, 4, 5]
    return (
        clean_data(pop_df, cols_drop_ind, new_cols)
        .slice(5, pop_df.height - 6)
    )

def prep_poor_data(poor_df: pl.DataFrame, reference_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the census poverty data.
    
    Parameters:
        poor_df (pl.DataFrame): The dataframe to prepare.
        reference_df (pl.DataFrame): The reference dataframe which contains the vocabulary.
    """
    new_cols = ['province', 'county', 'district', 'area', 'poor_pop_n', 'poor_pop_y']
    cols_drop_ind = [0, 5, 6, 7, 9, 10]
    return (
        clean_data(poor_df, cols_drop_ind, new_cols)
        .slice(7, poor_df.height - 10)
        # I only care for total values in `area`
        .filter(pl.col('area').str.contains('Total'))
        .drop('area')
        .with_columns(
            [spell_column(reference_df, col, 95) for col in ['province', 'county', 'district']]
        )
    )

def prep_house_appliances_data(house_appliances_df: pl.DataFrame, reference_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the census house appliances data.
    
    Parameters:
        house_appliances_df (pl.DataFrame): The dataframe to prepare.
        reference_df (pl.DataFrame): The reference dataframe which contains the vocabulary.
    """
    new_cols = ['province', 'county', 'area', 'refrigerator_y', 'refrigerator_n', 'washing_machine_y', 'washing_machine_n', 'dryer_y', 'dryer_n', 'micro_y', 'micro_n', 'extractor_y', 'extractor_n', 'car_y', 'car_n', 'moto_y', 'moto_n']
    cols_drop_ind = [0]
    return (
        clean_data(house_appliances_df, cols_drop_ind, new_cols)
        .slice(6)
        .filter(pl.col('area').str.contains('Total'))
        .drop('area')
        .with_columns(
            [spell_column(reference_df, col, 95) for col in ['province', 'county']]
        )
    )

def prep_house_tic_data(house_tic_df: pl.DataFrame, reference_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the census house tic data.
    
    Parameters:
        house_tic_df (pl.DataFrame): The dataframe to prepare.
        reference_df (pl.DataFrame): The reference dataframe which contains the vocabulary.
    """
    new_cols = ['province', 'county', 'district', 'tel_y', 'tel_n', 'cel_y', 'cel_n', 'tv_y', 'tv_n', 'internet_y', 'internet_n', 'computer_y', 'computer_n']
    cols_drop_ind = [0]
    return (
        clean_data(house_tic_df, cols_drop_ind, new_cols)
        .slice(6)
        .filter(~pl.col('district').str.contains('Total'))
        .with_columns(
            [spell_column(reference_df, col, 95) for col in ['province', 'county', 'district']]
        )
    )

def merge_data(pop_df: pl.DataFrame, poor_df: pl.DataFrame, house_appliances_df: pl.DataFrame, house_tic_df: pl.DataFrame) -> pl.DataFrame:
    """Merge the prepared census datasets.
    
    Parameters:
        pop_df (pl.DataFrame): The prepared population dataframe.
        poor_df (pl.DataFrame): The prepared poverty dataframe.
        house_appliances_df (pl.DataFrame): The prepared house appliances dataframe.
        house_tic_df (pl.DataFrame): The prepared house tic dataframe.
    """
    logging.info("Datasets prepared. Ready for merge.")
    return (
        pop_df
        .join(poor_df, on=['province', 'county', 'district'], how='left')
        .join(house_appliances_df, on=['province', 'county'], how='left')
        .join(house_tic_df, on=['province', 'county', 'district'], how='left')
        .with_columns(
            pl.all().exclude(['province', 'county', 'district']).cast(pl.Float64())
        )
        # After correct speling, some duplicates are created
        .group_by(['province', 'county', 'district'])
        .mean() # Ensure one row for each location
    )

if __name__ == "__main__":
    pop_df, poor_df, house_appliances_df, house_tic_df = load_data()
    # I will use the population data locations as vocabulary reference for fuzzy matching.
    reference_df = prep_pop_data(pop_df)
    merge_data(
        reference_df,
        prep_poor_data(poor_df, reference_df),
        prep_house_appliances_data(house_appliances_df, reference_df),
        prep_house_tic_data(house_tic_df, reference_df) 
    ).write_csv(CENSUS_PATH / CENSUS_FILE)
    logging.info(f"Data saved to {CENSUS_PATH / CENSUS_FILE}")
