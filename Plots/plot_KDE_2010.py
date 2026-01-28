import pandas as pd
import numpy as np
import pygmt
import xarray as xr
import os

# =============================================================================
# 1. PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'Figuras_paper')


os.makedirs(FIG_DIR, exist_ok=True)



trench_path = os.path.join(DATA_DIR, 'trench-chile')


peaks_all_path = os.path.join(DATA_DIR, 'peaks_2010.csv')
peaks_kde_csv_path = os.path.join(DATA_DIR, 'peaks_2010_kde.csv')


kde_grd_path = os.path.join(DATA_DIR, 'kde_2010.grd')


topo_nc = os.path.join(DATA_DIR, 'chile_central.nc')
topo_int = os.path.join(DATA_DIR, 'chile_central.int')
cpt_path = os.path.join(DATA_DIR, 'grayscale02.cpt')


# =============================================================================
# 3. LOAD DATA
# =============================================================================


trench_chile = pd.read_csv(
    trench_path,
    delim_whitespace=True,
    header=None,
    names=['Longitude', 'Latitude', 'Depth']
)

peaks_df_2010_all = pd.read_csv(peaks_all_path, index_col=False)
peaks_df_2010_kde = pd.read_csv(peaks_kde_csv_path, index_col=False)


kde_da = pygmt.load_dataarray(kde_grd_path)


kde_da_filt = kde_da.where(kde_da > 0)

# Hypocenter 2010
hypo_lat = -36.29
hypo_lon = -73.239
hypo_depth = 30.1

# =============================================================================
# 4. GRAPHICS (FIGURE S1)
# =============================================================================

region = [-76, -70, -39, -33]
projection = 'M6.5c'

fig1 = pygmt.Figure()

fig1.basemap(region=region, projection=projection, frame=['WSne', 'xa2f1', 'ya2f1'])


pygmt.makecpt(cmap=cpt_path, series=[-30000, 6000, 10])
fig1.grdimage(
    grid=topo_nc,
    projection=projection,
    shading=topo_int,
    cmap=True
)


pygmt.makecpt(cmap='hot', series=[0, 0.1], reverse=True)
fig1.grdimage(
    grid=kde_da_filt,
    projection=projection,
    cmap=True,
    nan_transparent=True
)


fig1.plot(x=peaks_df_2010_all['lon'], y=peaks_df_2010_all['lat'], fill='green', style='c0.1c', pen="0.05p,black", label="Model peaks+N3")
fig1.plot(x=peaks_df_2010_kde['lon'], y=peaks_df_2010_kde['lat'], fill='blue', style='x0.2c', pen="1.5p,blue", label="Peak (KDE)")
fig1.plot(x=hypo_lon, y=hypo_lat, fill='cyan', style='a0.2c', pen="0.5p,black", label="Hypocenter")


fig1.coast(shorelines=True, projection=projection)
fig1.plot(
    x=trench_chile['Longitude'],
    y=trench_chile['Latitude'],
    style="f0.5i/0.05i+r+t",
    fill='black',
    pen="0.5p,black"
)


fig1.colorbar(frame='af+lKDE density', position="JBC+o0.0c/0.9c+w6c/0.4c+h")
fig1.legend(position="JTC+o0.0c/0.1c+w8c")


save_path = os.path.join(FIG_DIR, 'FigS1.pdf')
fig1.savefig(save_path)
print(f"Figure saved at: {save_path}")

# fig1.show()