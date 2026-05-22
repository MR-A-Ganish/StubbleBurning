import urllib.request
import os
from urllib.error import HTTPError, URLError

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "india_districts.geojson":
        "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",

    "india_states.geojson":
        "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
}

for filename, url in FILES.items():
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"{filename} already exists ✔")
        continue

    try:
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filepath)
        print("Done ✅")
    except HTTPError as e:
        print(f"HTTP Error ❌ {e.code} for {filename}")
    except URLError as e:
        print(f"URL Error ❌ {e.reason}")
    except Exception as e:
        print(f"Unexpected error ❌ {e}")

print("\nAll available files are ready.")
