from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

from image_wrapper import ImageWrapper
image_wrapper = ImageWrapper()

DATAFRAME_PATH = "dataset.parquet"
RAW_DATA_PATH = "raw_data"


parquets = [pd.read_parquet(parquet) for parquet in tqdm(Path("parquet-data").glob("*.parquet"))]
dataframe = pd.concat(parquets, ignore_index=True)

metadata = []
for idx, item in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Appending MetaData"):
    try:
        if item['format'] == 'image':
            metadata.append(
                image_wrapper.fetch_latlong_and_datetime(Path("raw_data").joinpath(item['id'] + ".jpg")))
        else:
            metadata.append({'datetime': None, 'latitude': None, 'longitude': None})
    except:
        metadata.append({'datetime': None, 'latitude': None, 'longitude': None})


final_metadata = []
for meta in metadata:
    if 'datetime' not in meta: print("datetime not found")
    if 'latitude' not in meta: print("latitude not found")
    if 'longitude' not in meta: print("longitude not found")

    assert len(meta) == 3

    if meta['datetime']:
        meta['datetime'] = meta['datetime'].values
    
    final_metadata.append(json.dumps(meta))

dataframe['metadata'] = final_metadata
dataframe.to_parquet(DATAFRAME_PATH)