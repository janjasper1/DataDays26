import geopandas as gpd
from shapely.wkt import loads
import pandas as pd
from pathlib import Path


BOUNDARY_FILE = "OSMB-07946053ba77766c423655ac17c41565bf60514d.geojson"

def clean_data(df):
    # fill NaN capacity with 0
    df['capacity'] = df['capacity'].fillna(0)

    # drop columns with cars, scooters, cargo bikes, and mopeds
    columns_to_drop = ['num_cars_available', 'num_scooters_available', 'num_cargo_bicycles_available',
                       'num_mopeds_available', 'rental_uris_android', 'rental_uris_ios', 'rental_uris_web']
    df = df.drop(columns=columns_to_drop)

    # drop rows outside mannheim
    # 1. WKT-Strings in echte Geometrie-Objekte umwandeln (behebt deinen TypeError)
    df['geometry'] = df['geometry'].apply(lambda x: loads(x) if isinstance(x, str) else x)

    # 2. In ein GeoDataFrame umwandeln
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    # 3. Mannheim-Grenze laden
    boundary_path = Path(__file__).resolve().parent / BOUNDARY_FILE
    mannheim_boundary = gpd.read_file(boundary_path)

    # 4. Koordinatensysteme abgleichen
    mannheim_boundary = mannheim_boundary.to_crs(gdf.crs)

    # 5. Räumlicher Filter: Nur Stationen innerhalb der Grenze behalten
    df_mannheim = gpd.sjoin(gdf, mannheim_boundary, predicate='within')

    # 6. Aufräumen: Spalten vom Join entfernen, die wir nicht brauchen
    df_mannheim = df_mannheim.drop(columns=['index_right'], errors='ignore')

    ### prepare set for training
    df_ml = df_mannheim.copy()
    # handle geometry objects
    if 'geometry' in df_ml.columns:
        df_ml['Laengengrad'] = df_ml.geometry.x
        df_ml['Breitengrad'] = df_ml.geometry.y
        df_ml = df_ml.drop(columns=['geometry'])

    # handle time objects
    df_ml['last_reported'] = pd.to_datetime(df_ml['last_reported'], errors='coerce')

    df_ml['Stunde'] = df_ml['last_reported'].dt.hour
    df_ml['Wochentag'] = df_ml['last_reported'].dt.dayofweek
    df_ml['Monat'] = df_ml['last_reported'].dt.month

    df_ml = df_ml.drop(columns=['last_reported'])

    # handle boolean values:
    mapping = {
        'True': 1, 'False': 0,
        'true': 1, 'false': 0,
        'T': 1, 'F': 0,
        't': 1, 'f': 0,
        '1': 1, '0': 0,
        True: 1, False: 0,
        1: 1, 0: 0
    }

    # Wir nutzen .map() statt .replace(), das verhindert die FutureWarnings
    df_ml['is_virtual_station'] = df_ml['is_virtual_station'].map(mapping).astype(float)
    df_ml['realtime_data_outdated'] = df_ml['realtime_data_outdated'].map(mapping).astype(float)

    # remove unnecessary data
    df_ml = df_ml.rename(columns={'name_left': 'name'})
    df_ml = df_ml.drop(columns=['FID', 'feed_id', 'datastore_updated_at'])
    osm_muell = [
        'osm_id', 'name_right', 'name_en', 'boundary', 'admin_level',
        'admin_centre_node_id', 'admin_centre_node_lat', 'admin_centre_node_lng',
        'label_node_id', 'label_node_lat', 'label_node_lng'
    ]

    # ... und weg damit!
    df_ml = df_ml.drop(columns=osm_muell, errors='ignore')

    return df_ml

