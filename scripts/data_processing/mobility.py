import gc
from tqdm import tqdm

import polars as pl
import h3

from scripts import logging
from scripts import RAW_DATA_DIR, PROCESSED_DATA_DIR, MOBILITY_FILE, AGG_MOBILITY_FILE, RESOLUTION

def add_hex(lat_col: str, lon_col: str, resolution: int) -> pl.Expr:
    return (
        pl.struct([lat_col, lon_col]).map_elements(
            lambda v: h3.latlng_to_cell(v.get(lat_col), v.get(lon_col), resolution),
            return_dtype=pl.String
        ).alias('hex_id')
    )

def process_raw_data() -> pl.Expr:
    return (
        add_hex('lat', 'lon', resolution=RESOLUTION),
        pl.from_epoch('timestamp', 's')
    )

def aggregate(ms):   
    # Number of distinct devices by hex
    devices_count_by_hex = (
        ms.select(['hex_id', 'device_id'])
        .group_by('hex_id')
        .n_unique().rename({'device_id': 'devices_count'})
    )

    # Median duration by hex
    duration_by_hex = (
        # Sort by timestamp
        ms.sort('timestamp')
        # Remove consecutive rows with the same `hex_id`, only keep the first one
        .with_columns(
            pl.col('hex_id').shift(1).over('device_id').fill_null('').alias('prev_hex'),
            pl.col('timestamp').last().over('device_id').eq(pl.col('timestamp')).alias('is_last')
        ) 
        .filter(
            (pl.col('hex_id') != pl.col('prev_hex')) | (pl.col('is_last'))
        ).drop(['prev_hex', 'is_last']) 
        # Compute duration (first difference `timestamp`) and shift to allocate to the
        # right `hex_id`
        .with_columns(
            pl.col('timestamp').diff().shift(-1).dt.total_seconds().over('device_id').alias('duration_s')
        ) 
        # Drop nulls duration (the last registry of each `device_id`)
        .drop_nulls().select(['hex_id', 'duration_s'])
        # Calculate the median over each `hex_id`
        .group_by('hex_id').median()
    )
    # Join aggregated features and return. Left join to ensure all the hex_ids are returned.
    return devices_count_by_hex.join(duration_by_hex, on='hex_id', how='left')
    
def batch_agg(df, agg_function, col: str, batch_size: int, output_path: str):   
    logging.info("Starting data aggregation.")
    col_uniques = df.select(pl.col(col).unique()).collect()
    
    # Ensure the file is empty before writing the first batch
    first_write = True  
    
    for i in tqdm(range(0, len(col_uniques), batch_size), desc="Aggregating Data", unit="batch"):
        df_batch = df.filter(pl.col(col).is_in(col_uniques[i:i+batch_size]))
        batch_result = agg_function(df_batch)  # Perform aggregation
    
        # Write to CSV (append after first batch)
        with open(output_path, "a" if not first_write else "w", newline="") as f:
            batch_result.collect().write_csv(f, include_header=first_write)
        first_write = False  # After first batch, only append

        del batch_result
        gc.collect()
    
    logging.info(f"Aggregation completed. Results saved to {output_path}")


if __name__ == '__main__':
    # Load mobility data
    mobility_scan = pl.scan_parquet(RAW_DATA_DIR / MOBILITY_FILE)

    # Process raw data
    mobility_data = mobility_scan.with_columns(process_raw_data())

    # Save processed data
    mobility_data.collect().write_parquet(PROCESSED_DATA_DIR / MOBILITY_FILE)

    # Load processed data
    mobility_data = pl.scan_csv(PROCESSED_DATA_DIR / MOBILITY_FILE)
    # Aggregate mobility data and save 
    batch_agg(
        df= mobility_data, # Collect the LazyFrame to get the DataFrame
        agg_function= aggregate,
        col= 'hex_id',
        batch_size= 50,
        output_path= PROCESSED_DATA_DIR / AGG_MOBILITY_FILE
    )
