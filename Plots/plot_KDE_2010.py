import pandas as pd
import numpy as np
import pygmt
import sys
import xarray as xr
sys.path.append('/home/jtommy/Escritorio/Respaldo/functions/') 


slab_soam = pygmt.load_dataarray('/home/jtommy/Escritorio/Respaldo/base_de_datos/slab2/grid/sam_slab2_dep_02.23.18.grd') 

trench_chile = pd.read_csv(
    '/home/jtommy/Escritorio/Respaldo/base_de_datos/Trench/trench-chile',
    delim_whitespace=True,
    header=None,
    names=['Longitude', 'Latitude', 'Depth']
)


peaks_df_2010_all = pd.read_csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/asperities_peaks/peaks_2010.csv', index_col=False)
peaks_df_2010_kde = pd.read_csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/asperities_peaks/peaks_2010_kde.csv', index_col=False)

# Kernel Density 2010 asperity peaks

kde_da = pygmt.load_dataarray('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/asperities_peaks/kde_2010.grd')

# 2010 Hypocenter
hypo_lat = -36.29
hypo_lon = -73.239
hypo_depth = 30.1

#### MAP ####
region = [-76, -70, -39, -33]
projection = 'M6.5c'


kde_da_filt = kde_da.where(kde_da > 0)


fig1 = pygmt.Figure()


fig1.basemap(region=region, projection=projection, frame=['WSne', 'xa2f1', 'ya2f1'])

pygmt.makecpt(cmap='/home/jtommy/Escritorio/graficos_GMT/paletas/grayscale02.cpt', series=[-30000, 6000, 10])
fig1.grdimage(
    grid='/home/jtommy/Escritorio/Respaldo/base_de_datos/Topobatimetrias/chile_central.nc',
    projection=projection,
    shading='/home/jtommy/Escritorio/Respaldo/base_de_datos/Topobatimetrias/chile_central.int',
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


fig1.savefig('/home/jtommy/Escritorio/Respaldo/Paper2_v2/Figuras_paper/FigS1.pdf')
fig1.show()