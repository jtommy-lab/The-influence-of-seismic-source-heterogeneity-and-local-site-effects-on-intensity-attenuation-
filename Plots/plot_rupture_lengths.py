import pandas as pd
import numpy as np
import pygmt
import os
import sys

# =============================================================================
# 1. PATHS
# =============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')


FIG_DIR = os.path.join(PROJECT_ROOT, 'Figuras_paper')
os.makedirs(FIG_DIR, exist_ok=True) 

# =============================================================================
# 2. LOAD DATA
# =============================================================================

damage_path = os.path.join(DATA_DIR, 'daños_historicos_final_dataset.csv')
trench_path = os.path.join(DATA_DIR, 'trench-chile')
grid_path   = os.path.join(DATA_DIR, 'chile_central.nc')
shading_path= os.path.join(DATA_DIR, 'chile_central.int')
cpt_path    = os.path.join(DATA_DIR, 'grayscale02.cpt')


trench_chile = pd.read_csv(trench_path, delim_whitespace=True, index_col=None, names=['Longitude','Latitude','Depth'])

daños = pd.read_csv(damage_path, index_col=None)
daños = daños.loc[daños['Period'] == -1]
daños = daños.loc[daños['Rhyp [km]'] < 600]
lat_sites = daños['Latitude'].values
lon_sites = daños['Longitude'].values

# =============================================================================
# 3. GRAPHICS
# =============================================================================

fig1 = pygmt.Figure()
region = [-81, -69, -43, -27]
projection_log = "X?l/?"
projection = "M8c"

fig1.basemap(region=region, projection=projection, frame=['WSne', 'xa3f1', 'ya2f1'])


pygmt.makecpt(cmap=cpt_path, series=[-30000, 6000, 10])


fig1.grdimage(grid=grid_path, projection=projection, shading=shading_path, cmap=True)

fig1.coast(shorelines=True, projection=projection)
fig1.plot(x=lon_sites, y=lat_sites, style='t0.25c', fill='green', pen='0.1p,black', projection=projection, label="MSK64 intensities")
fig1.legend(position="JTL+jTL+o0.2c", box='+gwhite+p1p')


fig1.plot(x=[-75, -75], y=[-38, -34], pen='2p,red', projection=projection)
fig1.text(x=-75.31, y=-36, text="2010", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90) 

fig1.plot(x=[-75 -0.018*(2010-1835), -75 -0.018*(2010-1835)], y=[-37.8, -35.2], pen='2p,blue', projection=projection)
fig1.text(x=-75 -0.018*(2010-1835)-0.31, y=-36.5, text="1835", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)  

fig1.plot(x=[-75 -0.018*(2010-1928), -75 -0.018*(2010-1928)], y=[-35.5, -34.2], pen='2p,black', projection=projection)
fig1.text(x=-75 -0.018*(2010-1928)-0.31, y=-34.85, text="1928", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)  

fig1.plot(x=[-75 -0.018*(2010-1971), -75 -0.018*(2010-1971)], y=[-31.2, -32.3], pen='2p,black', projection=projection)
fig1.text(x=-75 -0.018*(2010-1971)+0.31, y=-31.75, text="1971", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)  

fig1.plot(x=[-75 -0.018*(2010-1751), -75 -0.018*(2010-1751)], y=[-38.2, -34], pen='2p,blue', projection=projection)
fig1.text(x=-75 -0.018*(2010-1751)+0.31, y=-36.1, text="1751", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90) 

fig1.plot(x=[-75 -0.018*(2010-1730), -75 -0.018*(2010-1730)], y=[-36.4, -29.6], pen='2p,blue', projection=projection)
fig1.text(x=-75 -0.018*(2010-1730)-0.31, y=-33.5, text="1730", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)  

fig1.plot(x=[-75 -0.018*(2010-2015), -75 -0.018*(2010-2015)], y=[-32, -30], pen='2p,red', projection=projection)
fig1.text(x=-75 -0.018*(2010-2015)+0.31, y=-31, text="2015", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)

fig1.plot(x=[-75 -0.018*(2010-1943), -75 -0.018*(2010-1943)], y=[-32, -30], pen='2p,black', projection=projection)
fig1.text(x=-75 -0.018*(2010-1943)-0.31, y=-31, text="1943", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)

fig1.plot(x=[-75 -0.018*(2010-1880), -75 -0.018*(2010-1880)], y=[-32, -30], pen='2p,black', projection=projection)
fig1.text(x=-75 -0.018*(2010-1880)-0.31, y=-31, text="1880", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)

fig1.plot(x=[-75 -0.018*(2010-1985), -75 -0.018*(2010-1985)], y=[-32.5, -34.3], pen='2p,red', projection=projection)
fig1.text(x=-75 -0.018*(2010-1985)-0.31, y=-33.4, text="1985", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)

fig1.plot(x=[-75 -0.018*(2010-1906), -75 -0.018*(2010-1906)], y=[-32.5, -34.3], pen='2p,blue', projection=projection)
fig1.text(x=-75 -0.018*(2010-1906)-0.31, y=-33.4, text="1906", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)

fig1.plot(x=[-75 -0.018*(2010-1822), -75 -0.018*(2010-1822)], y=[-32.5, -34.3], pen='2p,black', projection=projection)
fig1.text(x=-75 -0.018*(2010-1822)-0.31, y=-33.4, text="1822", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=90)


fig1.plot(x=trench_chile['Longitude'], y=trench_chile['Latitude'], style="f0.5i/0.05i+r+t", fill='black', pen="1p,black", projection=projection)


fig1.plot(x=[-72.4, -70.34], y=[-29.8, -29.98], pen='2p,black', projection=projection) 
fig1.plot(x=[-70.55, -70.34], y=[-31.98, -29.95], pen='2p,black', projection=projection) 
fig1.text(x=-71.5, y=-30.7, text="VP1", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=-5)
fig1.plot(x=[-72.62, -70.55], y=[-31.4, -31.6], pen='2p,black', projection=projection) 
fig1.plot(x=[-70.89,  -70.55], y=[-33.42, -31.98], pen='2p,black', projection=projection) 
fig1.text(x=-71.6, y=-32.3, text="VP2", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=-10)
fig1.plot(x=[-72.77, -70.89], y=[-32.9, -33.45], pen='2p,black', projection=projection) 
fig1.plot(x=[-70.89, -71.26], y=[-33.42, -34.46], pen='2p,black', projection=projection) 
fig1.text(x=-72, y=-33.6, text="VP3", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=-22)
fig1.plot(x=[-73, -71.26], y=[-33.8, -34.46], pen='2p,black', projection=projection) 
fig1.plot(x=[-71.26,-72.05], y=[-34.43,-36.16], pen='2p,black', projection=projection) 
fig1.text(x=-72.5, y=-35.1, text="M1", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=-22) 
fig1.plot(x=[-73.8, -72.05], y=[-35.6, -36.16], pen='2p,black', projection=projection) 
fig1.plot(x=[-72.05, -72.33], y=[-36.16, -36.87], pen='2p,black', projection=projection) 
fig1.text(x=-73.1, y=-36.25, text="M2", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=-20) 
fig1.plot(x=[-74.25, -72.33], y=[-36.5, -36.87], pen='2p,black', projection=projection) 
fig1.plot(x=[-72.33, -72.57], y=[-36.87, -37.8], pen='2p,black', projection=projection) 
fig1.text(x=-73.5, y=-37.15, text="M3", font="8p,Helvetica,black", projection=projection, fill="white", pen="0.25p,black,solid", angle=-11) 
fig1.plot(x=[-72.57, -74.5], y=[-37.77, -37.6], pen='2p,black', projection=projection) 

fig1.show()

# =============================================================================
# 4. SAVE FIGURE
# =============================================================================
save_path = os.path.join(FIG_DIR, 'Fig1.pdf')
fig1.savefig(save_path, dpi=300)

print(f"Figure saved at: {save_path}")