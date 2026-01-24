import pandas as pd
import numpy as np
import pygmt

### Define paths and read damage data ###

daños_path = '/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_historicos_final_dataset.csv'


daños = pd.read_csv(daños_path,index_col = None)
daños = daños.loc[daños['Period'] == -1]
daños = daños.loc[daños['Rhyp [km]'] < 600]
calibration_dataset = daños.loc[daños['Year']>=1985]
validation_dataset = daños.loc[daños['Year'] < 1985]


fig1 = pygmt.Figure()
projection_log = "X?l/?"

with fig1.subplot(nrows = 2, ncols = 2, figsize=("12c","6c"),sharex=True,sharey=True):
    with fig1.set_panel(panel = [0,0]):
        fig1.basemap(region=[10,750,4,10], projection=projection_log, frame=['Wsne','xfg3','ya2fg3'])
        fig1.plot(x=validation_dataset['Rhyp [km]'], y=validation_dataset['Intensity'], style='t0.15c',  pen='0.1p,blue',label = 'Validation data+N2')
        fig1.plot(x=calibration_dataset['Rhyp [km]'], y=calibration_dataset['Intensity'], style='c0.1c', pen='0.1p,red',label = 'Calibration data')
        fig1.text(text="a) R@-hyp@-", position='TL', font="8p,Helvetica,black", projection=projection_log,fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1')
        fig1.legend(position="JBR+jBR+o-3.2c/-0.6c+w6c") 
    with fig1.set_panel(panel = [1,0]):
        fig1.basemap(region=[10,750,4,10], projection=projection_log, frame=['WSne','xa2fg3','ya2fg3'])
        fig1.plot(x=validation_dataset['Rasp max [km]'], y=validation_dataset['Intensity'], style='t0.15c', pen='0.1p,blue')
        fig1.plot(x=calibration_dataset['Rasp max [km]'], y=calibration_dataset['Intensity'], style='c0.1c',pen='0.1p,red')
        fig1.text(text="c) R@-asp@-@+max@+", position='TL', font="8p,Helvetica,black", projection=projection_log,fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1')
        fig1.text(text="Distance (km)", position='BR', font="10p,Helvetica,black", projection=projection_log,no_clip = True,angle = 0,offset='1.2/-1')  
        fig1.text(text="MSK-64 Intensity", position='TL', font="10p,Helvetica,black", projection=projection_log,no_clip = True,angle = 90,offset='-1.2/-1')  
    with fig1.set_panel(panel = [0,1]):
        fig1.basemap(region=[10,750,4,10], projection=projection_log, frame=['wsne','xa2fg3','ya2fg3'])
        fig1.plot(x=validation_dataset['Rasp [km]'], y=validation_dataset['Intensity'], style='t0.15c', pen='0.1p,blue')
        fig1.plot(x=calibration_dataset['Rasp [km]'], y=calibration_dataset['Intensity'], style='c0.1c',pen='0.1p,red')
        fig1.text(text="b) R@-asp@-", position='TL', font="8p,Helvetica,black", projection=projection_log,fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1')
    with fig1.set_panel(panel = [1,1]):
        fig1.basemap(region=[10,750,4,10], projection=projection_log, frame=['wSne','xa2fg3','ya2fg3'])
        fig1.plot(x=validation_dataset['Rasp pond slip [km]'], y=validation_dataset['Intensity'], style='t0.15c',  pen='0.1p,blue')
        fig1.plot(x=calibration_dataset['Rasp pond slip [km]'], y=calibration_dataset['Intensity'], style='c0.1c',  pen='0.1p,red')
        fig1.text(text="d) R@-asp@-@+pond@+", position='TL', font="8p,Helvetica,black", projection=projection_log,fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1')
        
fig1.savefig('/home/jtommy/Escritorio/Respaldo/Paper2_v2/Figuras_paper/Fig2.pdf',dpi = 300)