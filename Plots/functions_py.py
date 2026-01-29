import numpy as np
import pandas as pd
import pygmt 
import os

# =============================================================================
# 1. PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

def get_slab_filename(Model_Longitudes, Model_Latitudes):
    """
    Search for the appropriate slab2 filename based on model longitudes and latitudes.
    """
    csv_path = os.path.join(DATA_DIR, 'slab2_info.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Falta el archivo: {csv_path}")
        
    slab_info = pd.read_csv(csv_path)

    
    lons = np.array(Model_Longitudes).copy()
    lats = np.array(Model_Latitudes).copy()
    lons[lons < 0] = lons[lons < 0] + 360
    
    
    slab_match = slab_info.loc[
        (slab_info['Minimum Longitude'] <= np.min(lons)) & 
        (slab_info['Maximum Longitude'] >= np.max(lons)) &
        (slab_info['Minimum Latitude'] <= np.min(lats)) & 
        (slab_info['Maximum Latitude'] >= np.max(lats))
    ]
    
    

    return slab_match['Filename'].to_list()[0]


def get_slab_depth(Model_Longitudes, Model_Latitudes):
    """
    Calculate slab depths for given model longitudes and latitudes.
    """
    
    filename = get_slab_filename(Model_Longitudes, Model_Latitudes)
    grdfile = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(grdfile):
        raise FileNotFoundError(f"No existe el grid: {grdfile}")

    
    lons = np.array(Model_Longitudes).copy()
    lons[lons < 0] = lons[lons < 0] + 360
    
    Model_df = pd.DataFrame({'Longitudes': lons, 'Latitudes': Model_Latitudes})
    
   
    Model_depth = pygmt.grdtrack(grid=grdfile, points=Model_df, newcolname='Depth (km)', radius=True)
    
    
    Model_depth['Depth (km)'] = Model_depth['Depth (km)'] * -1
    
    return Model_depth

