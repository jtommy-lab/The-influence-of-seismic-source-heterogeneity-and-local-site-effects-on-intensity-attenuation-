import pandas as pd
import pygmt
import numpy as np
import sys
import os
import functions_py 

# =============================================================================
# 1. PATHS
# =============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Regression', 'Stats_results')
FIG_DIR = os.path.join(PROJECT_ROOT, 'Figuras_paper')


os.makedirs(FIG_DIR, exist_ok=True)


calibration_path = os.path.join(RESULTS_DIR, 'residuals_df_instrumental.csv')
validation_path = os.path.join(RESULTS_DIR, 'residuals_df_validation.csv')
coeff_path = os.path.join(RESULTS_DIR, 'coeff_metrics_MSK.csv')
dBe_path = os.path.join(RESULTS_DIR, 'inter_event_residual_MSK.csv')

cpt_path = os.path.join(DATA_DIR, 'grayscale02.cpt')
topo_grid = os.path.join(DATA_DIR, 'chile_central.nc')
topo_int = os.path.join(DATA_DIR, 'chile_central.int')
trench_path = os.path.join(DATA_DIR, 'trench-chile')

residuals_MSK = pd.read_csv(calibration_path)
residuals_MSK_validation = pd.read_csv(validation_path)
coeff_MSK = pd.read_csv(coeff_path,index_col = 0)
dBe_MSK = pd.read_csv(dBe_path,index_col= 0)
metrics = ['Rhyp','Rasp_max','Rasp','Rasp_pond']
metric_label = ["a) R@-hyp@-","b) R@-asp@-@+max@+","c) R@-asp@-","d) R@-asp@-@+pond@+"]
projection_log = "X?l/?"

# =============================================================================
# 3. PLOT CALIBRATION EVENTS (2015, 2010, 1985)
# =============================================================================

for year in pd.unique(residuals_MSK['EVENT']):
    data_MSK = residuals_MSK.loc[residuals_MSK['EVENT'] == year]
    min_metrics = min(data_MSK['Rhyp'].min(),data_MSK['Rasp_max'].min(),data_MSK['Rasp'].min(),data_MSK['Rasp_pond'].min())
    max_metrics = max(data_MSK['Rhyp'].max(),data_MSK['Rasp_max'].max(),data_MSK['Rasp'].max(),data_MSK['Rasp_pond'].max())
    x_min = min_metrics-10
    x_max = int(max_metrics+50)    
    fig = pygmt.Figure()
    cont = 0
    with fig.subplot(nrows = 2, ncols = 2, figsize=("12c","6c"),sharex=True,sharey=True):
        for metric in metrics:    
            c1 = coeff_MSK.loc['c1',metric]
            c1_upper = c1 + 1.96* coeff_MSK.loc['se_c1',metric]
            c1_lower = c1 - 1.96* coeff_MSK.loc['se_c1',metric]
            c2 = coeff_MSK.loc['c2',metric]
            c2_upper = c2 + 1.96* coeff_MSK.loc['se_c2',metric]
            c2_lower = c2 - 1.96* coeff_MSK.loc['se_c2',metric]
            sigma = coeff_MSK.loc['sigma',metric]            
            data_MSK = data_MSK.sort_values(by = metric)
            R = data_MSK[metric]
            MSK = data_MSK['MSK64']
            pred_median = c1 + c2 * np.log(R) + dBe_MSK.loc[year,metric]
            pred_upper_sigma = pred_median + sigma
            pred_lower_sigma = pred_median - sigma
            pred_upper = c1_upper + c2_upper*np.log(R) + dBe_MSK.loc[year,metric]
            pred_lower = c1_lower + c2_lower*np.log(R) + dBe_MSK.loc[year,metric]
            poly_upper = pd.DataFrame({'R':R,'pred':pred_upper})
            poly_lower = pd.DataFrame({'R':R[::-1],'pred':pred_lower[::-1]})
            polygon = pd.concat([poly_upper,poly_lower])
            ### Stats ###
            RMSE = round(np.sqrt(np.mean((pred_median-MSK)**2)),2)
            bias = np.mean(MSK-pred_median)
            ###### Plot #####
            if cont == 0:                
                y_min = round(min(MSK.min(),pred_lower.min()))-1.5
                y_max = round(max(MSK.max(),pred_upper.max()))+1.5            
            print(year,metric,RMSE)
            with fig.set_panel(panel = cont):
                if cont == 0:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['Wsne','xa2fg3','ya2fg3'])
                if cont == 1:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['wsne','xa2fg3','ya2fg3'])
                if cont == 2:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['WSne','xa2fg3','ya2fg3'])
                if cont == 3:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['wSne','xa2fg3','ya2fg3'])            
                fig.plot(data=polygon,fill='lightgray',pen='0.1p,gray', projection=projection_log)
                if cont == 0:
                    fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log,label = 'Observations+N3')        
                    fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log,label = 'Mean prediction')
                    fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log, label = 'Mean ±σ')
                    fig.legend(position="JBR+jBR+o-4.5c/-0.6c+w9c") 
                else:
                    fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log)        
                    fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log)
                    fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log)
                fig.plot(x=R,y=pred_lower_sigma,pen='1p,blue,--', projection=projection_log)
                fig.text(text=metric_label[cont], position='TL', font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1', 
                         projection=projection_log)
                fig.text(text='RMSE = '+str(RMSE), position='BL', font="8p,Helvetica,black", fill="white",offset='0.1/0.1', projection=projection_log)                  
                if cont == 2:
                    fig.text(text="Distance (km)", position='BR', font="10p,Helvetica,black",no_clip = True,angle = 0,offset='1.2/-1', projection=projection_log) 
                    fig.text(text="MSK-64 Intensity", position='TL', font="10p,Helvetica,black", no_clip = True,angle = 90,offset='-1.2/-1', projection=projection_log)          
            cont = cont + 1    
    save_path = os.path.join(FIG_DIR, 'pred_'+str(year)+'.pdf')
    fig.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


# =============================================================================
# 4. PLOT VALIDATION EVENTS (1730, 1751, 1835, 1906)
# =============================================================================

for year in pd.unique(residuals_MSK_validation['EVENT']):
    data_MSK = residuals_MSK_validation.loc[residuals_MSK_validation['EVENT'] == year]
    min_metrics = min(data_MSK['Rhyp'].min(),data_MSK['Rasp_max'].min(),data_MSK['Rasp'].min(),data_MSK['Rasp_pond'].min())
    max_metrics = max(data_MSK['Rhyp'].max(),data_MSK['Rasp_max'].max(),data_MSK['Rasp'].max(),data_MSK['Rasp_pond'].max())
    x_min = min_metrics-10
    x_max = int(max_metrics+50)    
    fig = pygmt.Figure()
    cont = 0
    with fig.subplot(nrows = 2, ncols = 2, figsize=("12c","6c"),sharex=True,sharey=True):
        for metric in metrics:    
            c1 = coeff_MSK.loc['c1',metric]
            c1_upper = c1 + 1.96* coeff_MSK.loc['se_c1',metric]
            c1_lower = c1 - 1.96* coeff_MSK.loc['se_c1',metric]
            c2 = coeff_MSK.loc['c2',metric]
            c2_upper = c2 + 1.96* coeff_MSK.loc['se_c2',metric]
            c2_lower = c2 - 1.96* coeff_MSK.loc['se_c2',metric]
            sigma = coeff_MSK.loc['sigma',metric]            
            data_MSK = data_MSK.sort_values(by = metric)
            R = data_MSK[metric]
            MSK = data_MSK['MSK64']
            pred_median = c1 + c2 * np.log(R) #+ dBe_MSK.loc[year,metric]
            pred_upper_sigma = pred_median + sigma
            pred_lower_sigma = pred_median - sigma
            pred_upper = c1_upper + c2_upper*np.log(R) #+ dBe_MSK.loc[year,metric]
            pred_lower = c1_lower + c2_lower*np.log(R) #+ dBe_MSK.loc[year,metric]
            poly_upper = pd.DataFrame({'R':R,'pred':pred_upper})
            poly_lower = pd.DataFrame({'R':R[::-1],'pred':pred_lower[::-1]})
            polygon = pd.concat([poly_upper,poly_lower])
            ### Stats ###
            RMSE = round(np.sqrt(np.mean((pred_median-MSK)**2)),2)
            bias = round(np.mean(MSK-pred_median),2)
            ubRMSE = round(np.sqrt(RMSE**2-bias**2),2)
            ###### Plot #####
            if cont == 0:                
                y_min = round(min(MSK.min(),pred_lower.min()))-1.5
                y_max = round(max(MSK.max(),pred_upper.max()))+1.5            
            print(cont,year,metric_label[cont],bias)
            print(x_min,x_max,y_min,y_max)
            with fig.set_panel(panel = cont):
                if cont == 0:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['Wsne','xa2fg3','ya2fg3'])
                if cont == 1:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['wsne','xa2fg3','ya2fg3'])
                if cont == 2:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['WSne','xa2fg3','ya2fg3'])
                if cont == 3:
                    fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['wSne','xa2fg3','ya2fg3'])            
                fig.plot(data=polygon,fill='lightgray',pen='0.1p,gray', projection=projection_log)
                if cont == 0:
                    fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log,label = 'Observations+N3')        
                    fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log,label = 'Mean prediction')
                    fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log, label = 'Mean ±σ')
                    fig.legend(position="JBR+jBR+o-4.5c/-0.6c+w9c") 
                else:
                    fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log)        
                    fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log)
                    fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log)
                fig.plot(x=R,y=pred_lower_sigma,pen='1p,blue,--', projection=projection_log)
                fig.text(text=metric_label[cont], position='TL', font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1', 
                         projection=projection_log)
                fig.text(text='ubRMSE = '+str(ubRMSE), position='BL', font="8p,Helvetica,black", fill="white",offset='0.1/0.1', projection=projection_log) 
                fig.text(text='bias = '+str(bias), position='BL', font="8p,Helvetica,black", fill="white",offset='2.5/0.1', projection=projection_log)                  
                if cont == 2:
                    fig.text(text="Distance (km)", position='BR', font="10p,Helvetica,black",no_clip = True,angle = 0,offset='1.2/-1', projection=projection_log) 
                    fig.text(text="MSK-64 Intensity", position='TL', font="10p,Helvetica,black", no_clip = True,angle = 90,offset='-1.2/-1', projection=projection_log)          
            cont = cont + 1
    
    save_path = os.path.join(FIG_DIR, 'pred_'+str(year)+'.pdf')
    fig.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    
    
    
# =============================================================================
# 5. 2010 LOCKING MODEL (Figure 6)
# =============================================================================

# Definir paths específicos (Asumiendo que están en Data)
int_lock_path = os.path.join(DATA_DIR, 'daños_2010_dataset_Felipe_lock.csv')
peaks_lock_path = os.path.join(DATA_DIR, 'peaks_lock_2010.csv')
peaks_kde_path = os.path.join(DATA_DIR, 'peaks_2010_kde.csv')
locking_grd_path = os.path.join(DATA_DIR, 'Chile_locking.grd')

intensity_MSK_2010 = pd.read_csv(int_lock_path, index_col=0)
trench_chile = pd.read_csv(trench_path, delim_whitespace=True, index_col=None, names=['Longitude','Latitude','Depth'])

peak_2010 = pd.read_csv(peaks_lock_path, index_col=0)
peak_2010['weight'] = peak_2010['z']/sum(peak_2010['z'])
# Usamos functions_py importado desde la carpeta relativa
peak_2010['Depth (km)'] = functions_py.get_slab_depth(peak_2010['lon'].values,peak_2010['lat'].values)['Depth (km)'].values
peak_2010['lon'] = peak_2010['lon']-360

peaks_df_2010_kde = pd.read_csv(peaks_kde_path, index_col=0)
peaks_df_2010_kde['lon'] = peaks_df_2010_kde['lon']-360
year = 2010

data_MSK = intensity_MSK_2010.copy()
min_metrics = min(data_MSK['Rasp_max'].min(),data_MSK['Rasp'].min(),data_MSK['Rasp_pond'].min())
max_metrics = max(data_MSK['Rasp_max'].max(),data_MSK['Rasp'].max(),data_MSK['Rasp_pond'].max())
x_min = min_metrics-10
x_max = int(max_metrics+50) 

metric_label = ["a) R@-asp@-@+max@+","b) R@-asp@-","c) R@-asp@-@+pond@+"]
metrics = ['Rhyp','Rasp_max','Rasp','Rasp_pond']
   
fig = pygmt.Figure()
cont = 0
with fig.subplot(nrows = 3, ncols = 1, figsize=("6c","10c"),sharex=True,sharey=True):
    for metric in metrics[1:]:    
        c1 = coeff_MSK.loc['c1',metric]
        c1_upper = c1 + 1.96* coeff_MSK.loc['se_c1',metric]
        c1_lower = c1 - 1.96* coeff_MSK.loc['se_c1',metric]
        c2 = coeff_MSK.loc['c2',metric]
        c2_upper = c2 + 1.96* coeff_MSK.loc['se_c2',metric]
        c2_lower = c2 - 1.96* coeff_MSK.loc['se_c2',metric]
        sigma = coeff_MSK.loc['sigma',metric]            
        data_MSK = data_MSK.sort_values(by = metric)
        R = data_MSK[metric]
        MSK = data_MSK['MSK']
        pred_median = c1 + c2 * np.log(R) #+ dBe_MSK.loc[year,metric]
        pred_upper_sigma = pred_median + sigma
        pred_lower_sigma = pred_median - sigma
        pred_upper = c1_upper + c2_upper*np.log(R) #+ dBe_MSK.loc[year,metric]
        pred_lower = c1_lower + c2_lower*np.log(R) #+ dBe_MSK.loc[year,metric]
        poly_upper = pd.DataFrame({'R':R,'pred':pred_upper})
        poly_lower = pd.DataFrame({'R':R[::-1],'pred':pred_lower[::-1]})
        polygon = pd.concat([poly_upper,poly_lower])
        ### Stats ###
        RMSE = round(np.sqrt(np.mean((pred_median-MSK)**2)),2)
        bias = round(np.mean(MSK-pred_median),2)
        ubRMSE = round(np.sqrt(RMSE**2-bias**2),2)
        ###### Plot #####
        if cont == 0:                
            y_min = round(min(MSK.min(),pred_lower.min()))-1.5
            y_max = round(max(MSK.max(),pred_upper.max()))+1.5            
        print(cont,year,metric_label[cont],bias)
        print(x_min,x_max,y_min,y_max)
        with fig.set_panel(panel = cont):      
            if cont == 2:
                fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['WSne','xa2fg3+lDistance [km]','ya2fg3'])
            elif cont == 1:
                fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['Wsne','xa2fg3','ya2fg3+lMSK-64 Intensity']) 
            else:
                fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['Wsne','xa2fg3','ya2fg3'])                        
            fig.plot(data=polygon,fill='lightgray',pen='0.1p,gray', projection=projection_log)
            if cont == 0:
                fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log,label = 'Observations+N3')        
                fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log,label = 'Mean prediction')
                fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log, label = 'Mean ±σ')
                fig.legend(position="JTR+jTR+o-1.5c/-0.8c+w9c") 
            else:
                fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log)        
                fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log)
                fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log)
            fig.plot(x=R,y=pred_lower_sigma,pen='1p,blue,--', projection=projection_log)
            fig.text(text=metric_label[cont], position='TL', font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1', 
                        projection=projection_log)
            fig.text(text='ubRMSE = '+str(ubRMSE), position='BL', font="8p,Helvetica,black", fill="white",offset='0.1/0.1', projection=projection_log) 
            fig.text(text='bias = '+str(bias), position='BL', font="8p,Helvetica,black", fill="white",offset='2.5/0.1', projection=projection_log)         
        cont = cont + 1
        
fig.shift_origin(xshift="8c",yshift="0.8c")

region = [-75, -72, -38, -34]
projection = 'M5.5c'
Hypocenter_lat_2010 = -35.98
Hypocenter_lon_2010 = -73.15

fig.basemap(region=region, projection=projection, frame=['WSne','xa2f1','ya2f1'])        
pygmt.makecpt(cmap=cpt_path, series = [-30000,6000,10])
fig.grdimage(grid=topo_grid, projection=projection, shading=topo_int, cmap=True)
pygmt.makecpt(cmap='hot',series = [0,1],reverse=True)
fig.grdimage(grid=locking_grd_path, projection=projection, cmap=True, nan_transparent=True) 
fig.plot(x=peak_2010['lon'], y=peak_2010['lat'], fill='green',style='t0.3c',projection = projection,pen="0.1p,black",label = 'Preseismic peaks')
fig.plot(x=peaks_df_2010_kde['lon'], y=peaks_df_2010_kde['lat'], fill='blue',style='t0.3c',projection = projection,pen="0.1p,black",label = 'Coseismic peaks')
fig.coast(shorelines=True,projection=projection)
fig.plot(x=trench_chile['Longitude'], y=trench_chile['Latitude'], style="f0.5i/0.05i+r+t",fill='black', pen="1p,black",projection = projection)
fig.plot(x=Hypocenter_lon_2010, y=Hypocenter_lat_2010, style='a0.35c', fill='cyan', pen='0.2p,black', projection=projection,label='Hypocenter') 
fig.plot(x=[-73, -71.26], y=[-33.8, -34.46], pen='2p,black',projection = projection) #M1 Norte
fig.plot(x=[-71.26,-72.05], y=[-34.43,-36.16], pen='2p,black',projection = projection) #M1 cierre
fig.text(x = -72.5,y = -35.1,text="M1",  font="8p,Helvetica,black", projection=projection,fill="white",pen = "0.25p,black,solid",angle=-22) 
fig.plot(x=[-73.8, -72.05], y=[-35.6, -36.16], pen='2p,black',projection = projection) #M1 Sur/M2 Norte
fig.plot(x=[-72.05, -72.33], y=[-36.16, -36.87], pen='2p,black',projection = projection) #M2 cierre
fig.text(x = -73.1,y = -36.35,text="M2",  font="8p,Helvetica,black", projection=projection,fill="white",pen = "0.25p,black,solid",angle=-20) 
fig.plot(x=[-74.25, -72.33], y=[-36.5, -36.87], pen='2p,black',projection = projection) #M2 Sur/M3 Norte
fig.plot(x=[-72.33, -72.57], y=[-36.87, -37.8], pen='2p,black',projection = projection) #M3 cierre
fig.text(x = -73.5,y = -37.15,text="M3",  font="8p,Helvetica,black", projection=projection,fill="white",pen = "0.25p,black,solid",angle=-11) 
fig.plot(x=[-72.57, -74.5], y=[-37.77, -37.6], pen='2p,black',projection = projection) #M3 Sur
fig.legend(position = 'JTL+jTL+o0.1c',box='+gwhite+p1p')
fig.colorbar(frame="xaf+lLocking degree",position="g-75/-38.5+w5.5c/0.4c+h",projection=projection,region=[-75, -72,-38,-34])
# fig.show()

save_path = os.path.join(FIG_DIR, '2010_lock_pred.pdf')
fig.savefig(save_path, dpi=300)
print(f"Saved: {save_path}")



# =============================================================================
# 6. 2010 HIGH FREQUENCY MODEL (Figure 7)
# =============================================================================

# Definir paths HF
coeff_hf_path = os.path.join(RESULTS_DIR, 'coeff_metrics_2010_Palo_0.4_3Hz.csv')
int_hf_path = os.path.join(DATA_DIR, 'daños_2010_dataset_Felipe_HF_Palo_0.4_3_Hz.csv')
int_rasp_path = os.path.join(DATA_DIR, 'daños_2010_dataset_Felipe.csv')
peak_hf_path = os.path.join(DATA_DIR, 'peaks_hf_palo_2010_0.4_3Hz.csv')
slip_grid_path = os.path.join(DATA_DIR, 'slip_total_epsl12.grd')

coeff_MSK = pd.read_csv(coeff_hf_path, index_col=0)

intensity_MSK_2010 = pd.read_csv(int_hf_path)
intensity_MSK_Rasp_2010 = pd.read_csv(int_rasp_path)
intensity_MSK_Rasp_2010 = intensity_MSK_Rasp_2010.loc[intensity_MSK_Rasp_2010['Period'] == 0]
intensity_MSK_2010['Rasp_max'] = intensity_MSK_Rasp_2010['Rasp max [km]'].values
trench_chile = pd.read_csv(trench_path, delim_whitespace=True, index_col=None, names=['Longitude','Latitude','Depth'])

peak_2010 = pd.read_csv(peak_hf_path, index_col=0)
peak_2010['weight'] = peak_2010['z']/sum(peak_2010['z'])
peak_2010['Depth (km)'] = functions_py.get_slab_depth(peak_2010['lon'].values,peak_2010['lat'].values)['Depth (km)'].values
peak_2010['lon'] = peak_2010['lon']-360

peaks_df_2010_kde = pd.read_csv(peaks_kde_path, index_col=0)
peaks_df_2010_kde['lon'] = peaks_df_2010_kde['lon']-360
year = 2010

data_MSK = intensity_MSK_2010.copy()
min_metrics = min(data_MSK['Rasp_max'].min(),data_MSK['Rhf'].min(),data_MSK['Rhf_pond'].min())
max_metrics = max(data_MSK['Rasp_max'].max(),data_MSK['Rhf'].max(),data_MSK['Rhf_pond'].max())
x_min = min_metrics-10
x_max = int(max_metrics+50) 

metrics = ['Rasp_max','Rhf','Rhf_pond']
metric_label = ["a) R@-asp@-@+max@+","b) R@-hf@-","c) R@-hf@-@+pond@+"]

   
fig = pygmt.Figure()
cont = 0
with fig.subplot(nrows = 3, ncols = 1, figsize=("6c","10c"),sharex=True,sharey=True):
    for metric in metrics:    
        c1 = coeff_MSK.loc['c1',metric]
        c1_upper = c1 + 1.96* coeff_MSK.loc['se_c1',metric]
        c1_lower = c1 - 1.96* coeff_MSK.loc['se_c1',metric]
        c2 = coeff_MSK.loc['c2',metric]
        c2_upper = c2 + 1.96* coeff_MSK.loc['se_c2',metric]
        c2_lower = c2 - 1.96* coeff_MSK.loc['se_c2',metric]
        sigma = coeff_MSK.loc['sigma',metric] 
        AIC = coeff_MSK.loc['AIC',metric]            
        data_MSK = data_MSK.sort_values(by = metric)
        R = data_MSK[metric]
        MSK = data_MSK['MSK']
        pred_median = c1 + c2 * np.log(R) #+ dBe_MSK.loc[year,metric]
        pred_upper_sigma = pred_median + sigma
        pred_lower_sigma = pred_median - sigma
        pred_upper = c1_upper + c2_upper*np.log(R) #+ dBe_MSK.loc[year,metric]
        pred_lower = c1_lower + c2_lower*np.log(R) #+ dBe_MSK.loc[year,metric]
        poly_upper = pd.DataFrame({'R':R,'pred':pred_upper})
        poly_lower = pd.DataFrame({'R':R[::-1],'pred':pred_lower[::-1]})
        polygon = pd.concat([poly_upper,poly_lower])
        ### Stats ###
        RMSE = round(np.sqrt(np.mean((pred_median-MSK)**2)),2)
        bias = round(np.mean(MSK-pred_median),2)
        ubRMSE = round(np.sqrt(RMSE**2-bias**2),2)
        ###### Plot #####
        if cont == 0:                
            y_min = round(min(MSK.min(),pred_lower.min()))-1.5
            y_max = round(max(MSK.max(),pred_upper.max()))+1.5            
        print(cont,year,metric_label[cont],bias)
        print(x_min,x_max,y_min,y_max)
        with fig.set_panel(panel = cont):      
            if cont == 2:
                fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['WSne','xa2fg3+lDistance [km]','ya2fg3'])
            elif cont == 1:
                fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['Wsne','xa2fg3','ya2fg3+lMSK-64 Intensity']) 
            else:
                fig.basemap(region=[x_min, x_max, y_min, y_max], projection=projection_log, frame=['Wsne','xa2fg3','ya2fg3'])                        
            fig.plot(data=polygon,fill='lightgray',pen='0.1p,gray', projection=projection_log)
            if cont == 0:
                fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log,label = 'Observations+N3')        
                fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log,label = 'Mean prediction')
                fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log, label = 'Mean ±σ')
                fig.legend(position="JTR+jTR+o-1.5c/-0.8c+w9c") 
            else:
                fig.plot(x = R,y = MSK,style = 'c0.1c',fill = 'red', projection=projection_log)        
                fig.plot(x=R,y=pred_median,pen='1p,black', projection=projection_log)
                fig.plot(x=R,y=pred_upper_sigma,pen='1p,blue,--', projection=projection_log)
            fig.plot(x=R,y=pred_lower_sigma,pen='1p,blue,--', projection=projection_log)
            fig.text(text=metric_label[cont], position='TL', font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid",offset='0.1/-0.1', 
                        projection=projection_log)
            fig.text(text='σ = '+str(round(sigma,3)), position='BL', font="8p,Helvetica,black", fill="white",offset='0.1/0.1', projection=projection_log) 
            fig.text(text='AIC = '+str(round(AIC)), position='BL', font="8p,Helvetica,black", fill="white",offset='1.5/0.1', projection=projection_log)      
        cont = cont + 1
        
fig.shift_origin(xshift="8c",yshift="0.5c")

region = [-75, -71.5, -38, -33.5]
projection = 'M6c'
Hypocenter_lat_2010 = -35.98
Hypocenter_lon_2010 = -73.15

fig.basemap(region=region, projection=projection, frame=['WSne','xa2f1','ya2f1'])        
pygmt.makecpt(cmap=cpt_path, series = [-30000,6000,10])
fig.grdimage(grid=topo_grid, projection=projection, shading=topo_int, cmap=True)
pygmt.makecpt(cmap='hot',series = [0,1],reverse=True)
#fig.grdimage(grid = '/home/jtommy/Escritorio/Respaldo/base_de_datos/Modelos_locking/Chile_locking.grd',projection = projection, cmap=True,nan_transparent = True ) 
fig.plot(x=-80, y=-40, fill='red',style='c0.1c', projection = projection,pen="0.15p,black",label = 'HF peaks')
fig.plot(x=peak_2010['lon'], y=peak_2010['lat'], fill=peak_2010['z'],style='c0.15c',cmap = True, projection = projection,pen="0.1p,black")
fig.plot(x=peaks_df_2010_kde['lon'], y=peaks_df_2010_kde['lat'], fill='blue',style='t0.3c',projection = projection,pen="0.1p,black",label = 'Coseismic peaks')
fig.coast(shorelines=True,projection=projection)
fig.plot(x=trench_chile['Longitude'], y=trench_chile['Latitude'], style="f0.5i/0.05i+r+t",fill='black', pen="1p,black",projection = projection)
fig.plot(x=Hypocenter_lon_2010, y=Hypocenter_lat_2010, style='a0.35c', fill='cyan', pen='0.2p,black', projection=projection,label='Hypocenter')
fig.plot(x = intensity_MSK_Rasp_2010['Longitud'], y = intensity_MSK_Rasp_2010['Latitud'], style='t0.25c', fill='green', pen='0.1p,black',projection = projection,label="MSK64 intensities")
fig.grdcontour(grid=slip_grid_path, annotation=4, levels=4) 
fig.legend(position = 'JTL+jTL+o0.1c',box='+gwhite+p1p')
fig.colorbar(frame="xaf+lNormalized energy",position="g-75/-38.5+w5.5c/0.4c+h",projection=projection,region=[-75, -72,-38,-34])
# fig.show()

save_path = os.path.join(FIG_DIR, '2010_hf_palo_0.4_3Hz_pred.pdf')
fig.savefig(save_path, dpi=300)
print(f"Saved: {save_path}")
        
