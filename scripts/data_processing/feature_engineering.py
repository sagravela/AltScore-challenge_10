import polars as pl
from geopy import Nominatim, distance

from scripts import logging
from scripts import FE_FILE, NOMINATIM_PATH, NOMINATIM_FILE, PROCESSED_DATA_DIR

def get_city_coordinates() -> tuple:
    """Get coordinates for Cuenca, Guayaquil and Quito. Returns a tuple of tuples (lat, lon)."""
    logging.info("Requesting Nominatim for city coordinates...")
    geocoder = Nominatim(user_agent="_")
    _, cuenca = geocoder.geocode("Cuenca, Ecuador")
    _, guayaquil = geocoder.geocode("Guayaquil, Ecuador")
    _, quito = geocoder.geocode("Quito, Ecuador")
    return cuenca, guayaquil, quito

def get_distance(city_coords : tuple) -> pl.Expr:
    """Calculate distance in kilometers between each point location and a tuple of coordinates (lat, lon). 
    Returns a polars expression.
    
    Parameters:
        city_coords (tuple): A tuple of coordinates (lat, lon).
    """
    return (
        pl.struct([pl.col('lat'), pl.col('lon')])
        .map_elements(
            lambda x: distance.distance((x['lat'], x['lon']), city_coords).km,
            return_dtype=pl.Float64()
        )
    )

def calculate_distances(df: pl.DataFrame) -> pl.DataFrame:
    """Add the distance in kilometers between each row location (given by their `lat` and `lon`) and Cuenca, Guayaquil and Quito."""
    logging.info("Calculating distances...")
    cuenca, guayaquil, quito = get_city_coordinates()
    return (
        df
        .with_columns(
            get_distance(cuenca).alias('cuenca_km'),
            get_distance(guayaquil).alias('guayaquil_km'),
            get_distance(quito).alias('quito_km')
        )
    )

def get_closest_city(df: pl.DataFrame) -> pl.DataFrame:
    """Add the closest city to each row."""
    logging.info("Adding feature 'closest_city'...")
    distances = df.drop(['hex_id', 'lat', 'lon'])
    closest_sity = (
        # Convert the distances to booleans where true indicates the closest city
        (distances == distances.min_horizontal())
        .with_columns(
            pl.when(pl.col('cuenca_km')).then(pl.lit('Cuenca'))
            .when(pl.col('guayaquil_km')).then(pl.lit('Guayaquil'))
            .otherwise(pl.lit('Quito'))
            .alias('closest_city')
        )
        .select('closest_city')
    )

    return pl.concat([df, closest_sity], how='horizontal')


if __name__ == '__main__':
    logging.info(f"Loading data from {NOMINATIM_PATH}.")
    nominatin_df = pl.read_csv(NOMINATIM_PATH / NOMINATIM_FILE, columns=['hex_id', 'lat', 'lon'])
    distances_df = calculate_distances(nominatin_df)
    get_closest_city(distances_df).write_csv(PROCESSED_DATA_DIR / FE_FILE)