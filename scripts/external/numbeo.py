from io import StringIO

import pandas as pd
import requests

from scripts import logging
from scripts import NUMBEO_PATH, NUMBEO_CITIES, NUMBEO_PRICES_FILE, NUMBEO_QUALITY_FILE

def get_prices(city: str):
    """Scrape the historical prices from Numbeo.
    
    Parameters:
        city (str): The city to scrape. Has to coincide with the city name on Numbeo.
    """
    url = f"https://www.numbeo.com/cost-of-living/city-history/in/{city}"
    logging.info(f"Scraping {url}")
    response = requests.get(url)
    tables = pd.read_html(StringIO(response.text))
    data = tables[1]
    for i, table in enumerate(tables[2:]):
        # Add prefixes to avoid column name conflicts
        table.columns = [f"{i+1}_{col}" if col != 'Year' else 'Year' for col in table.columns]
        data = data.merge(table, how='outer')
    data.insert(0, 'city', city)
    return data

def get_quality_of_life(city: str):
    """Scrape the quality of life from Numbeo.
    
    Parameters:
        city (str): The city to scrape. Has to coincide with the city name on Numbeo.
    """
    url = f"https://www.numbeo.com/quality-of-life/in/{city}"
    logging.info(f"Scraping {url}")
    response = requests.get(url)
    tables = pd.read_html(StringIO(response.text))
    # Find the table with the data (one with 10 rows)
    data = tables[[len(t) for t in tables].index(10)].copy().dropna()
    data.insert(0, 'city', city)
    return data.pivot(index='city', columns=0, values=1).reset_index()


if __name__ == '__main__':
    logging.info(f"Scraping data from Numbeo.")

    # Historical prices
    prices = map(get_prices, NUMBEO_CITIES)
    logging.info(f"Data saved to {NUMBEO_PATH / NUMBEO_PRICES_FILE}.")
    pd.concat(prices).to_csv(NUMBEO_PATH / NUMBEO_PRICES_FILE, index=False)

    # Quality of life
    quality_of_life = map(get_quality_of_life, NUMBEO_CITIES)
    pd.concat(quality_of_life).to_csv(NUMBEO_PATH / NUMBEO_QUALITY_FILE, index=False)
    logging.info(f"Data saved to {NUMBEO_PATH / NUMBEO_QUALITY_FILE}.")
