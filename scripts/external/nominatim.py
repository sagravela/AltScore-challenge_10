import polars as pl
import h3
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

from scripts import logging
from scripts import RAW_DATA_DIR, TRAIN_FILE, TEST_FILE, NOMINATIM_PATH, NOMINATIM_FILE


def add_latlon(hex_ids: list[str]) -> list[dict]:
    """Add latitude and longitude to a list of hex ids.

    Parameters:
        hex_ids (list[str]): A list of hex ids.
    """
    return [
        {
            'hex_id': hex_id, 
            'lat': lat,
            'lon': lon
        } for hex_id, (lat, lon) in map(lambda x:(x, h3.cell_to_latlng(x)), hex_ids)
    ]

def apply_geocode(locations: list[dict]) -> list[dict]:
    """Geocode a list of locations.
    
    Parameters:
        locations (list[dict]): A list of locations which contains `hex_id`, `lat` and `lon` keys.
    """
    logging.info(f"Geocoding {len(locations)} locations.")
    geolocator = Nominatim(user_agent="_")
    # Rate limit the geocoding requests to be nice to the service
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1.5, max_retries=5)
    for i, l in enumerate(tqdm(locations, desc="Geocoding", unit="location")):
        try:
            # Get the address from the geocode service
            address = geocode((l['lat'], l['lon']), exactly_one=True, zoom=15).raw.get('address')
            l.update(address)
        except Exception as e:
            logging.info(f"Error rised: {e}")
        # Write every 10 steps
        if i % 10 == 0:
            pl.DataFrame(locations).write_csv(NOMINATIM_PATH / NOMINATIM_FILE, index=False)
    return locations

if __name__ == '__main__':
    logging.info(f"Loading data from {RAW_DATA_DIR}.")
    raw_train = pl.read_csv(RAW_DATA_DIR / TRAIN_FILE)
    raw_test = pl.read_csv(RAW_DATA_DIR / TEST_FILE)

    logging.info("Preparing data...")
    hex_ids = pl.concat([raw_train['hex_id'], raw_test['hex_id']]).to_list()
    locations = add_latlon(hex_ids)
    addresses = apply_geocode(locations)
    logging.info(f"Addresses saved to {NOMINATIM_PATH / NOMINATIM_FILE}.")

