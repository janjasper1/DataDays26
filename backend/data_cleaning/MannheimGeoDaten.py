import pandas as pd
import geopandas as gpd
from shapely.wkt import loads


# 1. Lade dein Datenset (z.B. von Hugging Face oder lokal)
# Falls die Daten noch als CSV vorliegen:
df = pd.read_csv("deine_datendatei.csv") 

# 2. Geometrie-Spalte (WKT) in echte Point-Objekte umwandeln
# Deine Daten nutzen das Format: POINT (8.601469 49.504677)
df['geometry'] = df['geometry'].apply(loads)

# 3. Erstelle ein GeoDataFrame
# Wir nutzen EPSG:4326 (WGS84 Koordinatensystem)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# 4. Lade die offizielle Mannheim-Grenze (deine hochgeladene Datei)
mannheim_boundary = gpd.read_file("OSMB-07946053ba77766c423655ac17c41565bf60514d.geojson")

# Sicherstellen, dass beide das gleiche Koordinatensystem nutzen
mannheim_boundary = mannheim_boundary.to_crs(gdf.crs)

# 5. Räumlicher Filter (Point-in-Polygon)
# Wir behalten nur die Punkte, die sich INNERHALB der Stadtgrenze befinden
mannheim_stations_only = gpd.sjoin(gdf, mannheim_boundary, predicate='within')

# 6. Unnötige Spalten aus dem Join entfernen (optional)
# sjoin fügt Spalten der Grenz-Datei hinzu (wie index_right), die wir nicht brauchen
mannheim_stations_only = mannheim_stations_only.drop(columns=['index_right'], errors='ignore')

# 7. Ergebnis speichern
mannheim_stations_only.to_csv("mannheim_filtered_data.csv", index=False)

print(f"Filterung abgeschlossen. Verbleibende Datenpunkte: {len(mannheim_stations_only)}")