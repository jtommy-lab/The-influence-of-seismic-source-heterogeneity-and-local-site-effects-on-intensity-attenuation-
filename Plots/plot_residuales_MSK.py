import pandas as pd
import pygmt
import numpy as np
import os

# =============================================================================
# 1. PATHS
# =============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Regression', 'Stats_results')


FIG_DIR = os.path.join(PROJECT_ROOT, 'Figuras_paper')
os.makedirs(FIG_DIR, exist_ok=True) # Crea la carpeta si no existe

# =============================================================================
# 2. LOAD DATA
# =============================================================================


residual_path = os.path.join(RESULTS_DIR, 'residuals_df_instrumental.csv')


residuals_MSK = pd.read_csv(residual_path)

nbins = 6


Rhyp_MSK_bins = residuals_MSK.copy()
Rhyp_min = Rhyp_MSK_bins['Rhyp'].min()
Rhyp_max = Rhyp_MSK_bins['Rhyp'].max()
Rhyp_bin = np.logspace(np.log10(Rhyp_min),np.log10(Rhyp_max),nbins+1)
Rhyp_MSK_bins['bin'] = pd.cut(Rhyp_MSK_bins['Rhyp'], bins=Rhyp_bin)
Rhyp_MSK_bins['bin_mid'] = Rhyp_MSK_bins['bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notnull(x) else None)
n_data_Rhyp = Rhyp_MSK_bins.groupby(['bin_mid']).count()['residuals_Rhyp'].values
bin_average_Rhyp = pd.DataFrame({'x':Rhyp_MSK_bins.groupby(['bin_mid'])['residuals_Rhyp'].mean().index.values.to_numpy(),
                                         'y':Rhyp_MSK_bins.groupby(['bin_mid'])['residuals_Rhyp'].mean().values,
                                         'xerr':0,
                                         'yerr':Rhyp_MSK_bins.groupby(['bin_mid'])['residuals_Rhyp'].std().values/np.sqrt(n_data_Rhyp)})


Rasp_max_MSK_bins = residuals_MSK.copy()
Rasp_max_min = Rasp_max_MSK_bins['Rasp_max'].min()
Rasp_max_max = Rasp_max_MSK_bins['Rasp_max'].max()
Rasp_max_bin = np.logspace(np.log10(Rasp_max_min),np.log10(Rasp_max_max),nbins+1)
Rasp_max_MSK_bins['bin'] = pd.cut(Rasp_max_MSK_bins['Rasp_max'], bins=Rasp_max_bin)
Rasp_max_MSK_bins['bin_mid'] = Rasp_max_MSK_bins['bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notnull(x) else None)
n_data_Rasp_max = Rasp_max_MSK_bins.groupby(['bin_mid']).count()['residuals_Rasp_max'].values
bin_average_Rasp_max = pd.DataFrame({'x':Rasp_max_MSK_bins.groupby(['bin_mid'])['residuals_Rasp_max'].mean().index.values.to_numpy(),
                                         'y':Rasp_max_MSK_bins.groupby(['bin_mid'])['residuals_Rasp_max'].mean().values,
                                         'xerr':0,
                                         'yerr':Rasp_max_MSK_bins.groupby(['bin_mid'])['residuals_Rasp_max'].std().values/np.sqrt(n_data_Rasp_max)})


Rasp_MSK_bins = residuals_MSK.copy()
Rasp_min = Rasp_MSK_bins['Rasp'].min()
Rasp_max = Rasp_MSK_bins['Rasp'].max()
Rasp_bin = np.logspace(np.log10(Rasp_min),np.log10(Rasp_max),nbins+1)
Rasp_MSK_bins['bin'] = pd.cut(Rasp_MSK_bins['Rasp_max'], bins=Rasp_bin)
Rasp_MSK_bins['bin_mid'] = Rasp_MSK_bins['bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notnull(x) else None)
n_data_Rasp = Rasp_MSK_bins.groupby(['bin_mid']).count()['residuals_Rasp'].values
bin_average_Rasp = pd.DataFrame({'x':Rasp_MSK_bins.groupby(['bin_mid'])['residuals_Rasp'].mean().index.values.to_numpy(),
                                         'y':Rasp_MSK_bins.groupby(['bin_mid'])['residuals_Rasp'].mean().values,
                                         'xerr':0,
                                         'yerr':Rasp_MSK_bins.groupby(['bin_mid'])['residuals_Rasp'].std().values/np.sqrt(n_data_Rasp)})


Rasp_pond_MSK_bins = residuals_MSK.copy()
Rasp_pond_min = Rasp_pond_MSK_bins['Rasp_pond'].min()
Rasp_pond_max = Rasp_pond_MSK_bins['Rasp_pond'].max()
Rasp_pond_bin = np.logspace(np.log10(Rasp_pond_min),np.log10(Rasp_pond_max),nbins+1)
Rasp_pond_MSK_bins['bin'] = pd.cut(Rasp_pond_MSK_bins['Rasp_pond'], bins=Rasp_pond_bin)
Rasp_pond_MSK_bins['bin_mid'] = Rasp_pond_MSK_bins['bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notnull(x) else None)
n_data_Rasp_pond = Rasp_pond_MSK_bins.groupby(['bin_mid']).count()['residuals_Rasp_pond'].values
bin_average_Rasp_pond = pd.DataFrame({'x':Rasp_pond_MSK_bins.groupby(['bin_mid'])['residuals_Rasp_pond'].mean().index.values.to_numpy(),
                                         'y':Rasp_pond_MSK_bins.groupby(['bin_mid'])['residuals_Rasp_pond'].mean().values,
                                         'xerr':0,
                                         'yerr':Rasp_pond_MSK_bins.groupby(['bin_mid'])['residuals_Rasp_pond'].std().values/np.sqrt(n_data_Rasp_pond)})


zero_line = np.zeros((1000,2))
zero_line[:,0] = np.linspace(0.1,2000,len(zero_line))

projection_log = "X?l/?"
fig1 = pygmt.Figure()
with fig1.subplot(nrows = 2, ncols = 2, figsize=("12c","6c"),sharex='b',sharey='l'):
    with fig1.set_panel(panel = [0,0]):
        fig1.basemap(region=[30,400,-2.5,2.5], projection=projection_log, frame=['xa2f1g3','ya1f.5g1'])
        fig1.text(text='a) R@-hyp@-', position = 'TL',offset="0.1/-0.1", font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid", projection=projection_log)
        fig1.plot(x=Rhyp_MSK_bins['bin_mid'].to_numpy(),y=Rhyp_MSK_bins['residuals_Rhyp'].to_numpy(),style='c0.08c',fill='red', projection=projection_log)
        fig1.plot(x = zero_line[:,0],y = zero_line[:,1], pen='1.2p,black', projection=projection_log)
        fig1.plot(data = bin_average_Rhyp,error_bar='+p0.75p,blue',style='t0.25c',fill='blue',pen = "0.25p,black,solid", projection=projection_log)
    with fig1.set_panel(panel = [0,1]):
        fig1.basemap(region=[30,400,-2.5,2.5], projection=projection_log, frame=['xa2f1g3','ya1f.5g1'])
        fig1.text(text='b) R@-asp@-@+max@+', position = 'TL',offset="0.1/-0.1", font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid", projection=projection_log)
        fig1.plot(x=Rasp_max_MSK_bins['bin_mid'].to_numpy(),y=Rasp_max_MSK_bins['residuals_Rasp_max'].to_numpy(),style='c0.08c',fill='red', projection=projection_log)
        fig1.plot(x = zero_line[:,0],y = zero_line[:,1], pen='1.2p,black', projection=projection_log)
        fig1.plot(data = bin_average_Rasp_max,error_bar='+p0.75p,blue',style='t0.25c',fill='blue',pen = "0.25p,black,solid", projection=projection_log,label = 'Binned mean residuals and standard error')
        fig1.legend(position="JBR+jBR+o0.9c/-0.7c+w9c") 
    with fig1.set_panel(panel = [1,0]):
        fig1.basemap(region=[30,400,-2.5,2.5], projection=projection_log, frame=['xa2f1g3','ya1f.5g1'])
        fig1.text(text='c) R@-asp@-', position = 'TL',offset="0.1/-0.1", font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid", projection=projection_log)
        fig1.plot(x=Rasp_MSK_bins['bin_mid'].to_numpy(),y=Rasp_MSK_bins['residuals_Rasp'].to_numpy(),style='c0.08c',fill='red', projection=projection_log)
        fig1.plot(x = zero_line[:,0],y = zero_line[:,1], pen='1.2p,black', projection=projection_log)
        fig1.plot(data = bin_average_Rasp,error_bar='+p0.75p,blue',style='t0.25c',fill='blue',pen = "0.25p,black,solid", projection=projection_log)
        fig1.text(text="Distance [km]", position = 'BR', font="10p,Helvetica,black", projection=projection_log,no_clip = True,offset ="1.5/-0.9")  
        fig1.text(text="Intra-event residual", position = 'TL', font="10p,Helvetica,black", projection=projection_log,no_clip = True,offset ="-1.3/-1.1",angle = 90)  
    with fig1.set_panel(panel = [1,1]):
        fig1.basemap(region=[30,400,-2.5,2.5], projection=projection_log, frame=['xa2f1g3','ya1f.5g1'])
        fig1.text(text='d) R@-asp@-@+pond@+', position = 'TL',offset="0.1/-0.1", font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid", projection=projection_log)
        fig1.plot(x=Rasp_pond_MSK_bins['bin_mid'].to_numpy(),y=Rasp_pond_MSK_bins['residuals_Rasp_pond'].to_numpy(),style='c0.08c',fill='red', projection=projection_log)
        fig1.plot(x = zero_line[:,0],y = zero_line[:,1], pen='1.2p,black', projection=projection_log)
        fig1.plot(data = bin_average_Rasp_pond,error_bar='+p0.75p,blue',style='t0.25c',fill='blue',pen = "0.25p,black,solid", projection=projection_log)


save_path_3 = os.path.join(FIG_DIR, 'Fig3.pdf')
fig1.savefig(save_path_3)
print(f"Figure 3 saved at: {save_path_3}")



# =============================================================================
# 3. PROCESSING DATA (FIGURE 8 - f0 RESIDUALS)
# =============================================================================

# Defining paths
res_f0_path = os.path.join(RESULTS_DIR, 'residuals_df_2010_Palo_0.4_3Hz.csv')
coeff_path  = os.path.join(RESULTS_DIR, 'coeff_metrics_2010_Palo_0.4_3Hz_f0.csv')

# Loading data
residuals_MSK_f0 = pd.read_csv(res_f0_path, index_col=0)
residuals_MSK_f0 = residuals_MSK_f0.sort_values(by='f0')
coeff_MSK = pd.read_csv(coeff_path, index_col=0)

metrics = ['Rhf_max','Rhf','Rhf_pond']
metric_label = ["a) R@-hf@-@+max@+","b) R@-hf@-","c) R@-hf@-@+pond@+"]



projection_log = "X8cl/4c"
fig1 = pygmt.Figure()

f0_test_01 = np.logspace(np.log10(0.01),np.log10(0.99),100)
f0_test_5 = np.logspace(np.log10(4),np.log10(10),100)
pred_f0_01 = 0.5939221*np.log10(f0_test_01)
pred_f0_5 = -0.5695806*np.log10(f0_test_5)
zero_line = np.zeros((1000,2))
zero_line[:,0] = np.linspace(0.001,2000,len(zero_line))

fig1.basemap(region=[0.01,10,-1.2,1.2], projection=projection_log, frame=['xa1f1g3+lFrequency [Hz]','ya.5f.25g.5+lResiduals'])
fig1.plot(x=residuals_MSK_f0['f0'].values,y=residuals_MSK_f0['residuals_Rasp_max'].values,style='c0.08c',fill='red', projection=projection_log,label = 'R@-asp@-@+max@+ residuals')
fig1.plot(x = f0_test_01,y = pred_f0_01, pen='1.2p,blue', projection=projection_log,label = 'bias')
fig1.plot(x = f0_test_5,y = pred_f0_5, pen='1.2p,blue', projection=projection_log)
fig1.plot(x = zero_line[:,0],y = zero_line[:,1], pen='1.2p,black', projection=projection_log)
fig1.legend(position = 'JTL+jTL+o0.1c',box='+gwhite+p1p')


save_path_8 = os.path.join(FIG_DIR, 'Fig8.pdf')
fig1.savefig(save_path_8)
print(f"Figure 8 saved at: {save_path_8}")


# =============================================================================
# 4. PROCESSING DATA (FIGURE S2 - METRICS)
# =============================================================================


f0_test_01 = np.logspace(np.log10(0.01),np.log10(0.99),100)
f0_test_5 = np.logspace(np.log10(4),np.log10(10),100)

zero_line = np.zeros((1000,2))
zero_line[:,0] = np.linspace(0.001,2000,len(zero_line))

projection_log = "X?l/?"

fig1 = pygmt.Figure()
cont = 0
with fig1.subplot(nrows = 3, ncols = 1, figsize=("12c","9c"),sharex='b',sharey='l'):
    for metric in metrics:
        pred_f0_01 = coeff_MSK.loc['c3',metric]*np.log10(f0_test_01)
        pred_f0_5 = coeff_MSK.loc['c4',metric]*np.log10(f0_test_5)
        with fig1.set_panel(panel = [cont]):
            if cont == 0:
                fig1.basemap(region=[0.01,10,-1.2,1.2], projection=projection_log, frame=['Wsne','xa1f1g3+lFrequency [Hz]','ya.5f.25g.5'])
            if cont == 1:
                fig1.basemap(region=[0.01,10,-1.2,1.2], projection=projection_log, frame=['Wsne','xa1f1g3','ya.5f.25g.5+lResiduals'])
            if cont == 2:
                fig1.basemap(region=[0.01,10,-1.2,1.2], projection=projection_log, frame=['WSne','xa1f1g3+lFrequency [Hz]','ya.5f.25g.5+'])
            print(metric_label[cont])
            fig1.plot(x=residuals_MSK_f0['f0'].values,y=residuals_MSK_f0['residuals_'+str(metric)].values,style='c0.08c',fill='red', projection=projection_log)
            fig1.text(text=metric_label[cont], position = 'TL',offset="0.1/-0.1", font="8p,Helvetica,black",fill="white",pen = "0.25p,black,solid", projection=projection_log)
            fig1.plot(x = f0_test_01,y = pred_f0_01, pen='1.2p,blue', projection=projection_log)
            fig1.plot(x = f0_test_5,y = pred_f0_5, pen='1.2p,blue', projection=projection_log)
            fig1.plot(x = zero_line[:,0],y = zero_line[:,1], pen='1.2p,black', projection=projection_log)
            cont = cont+1

# SAVE FIGURE S2
save_path_S2 = os.path.join(FIG_DIR, 'FigS2.pdf')
fig1.savefig(save_path_S2)
print(f"Figure S2 saved at: {save_path_S2}")