# The-influence-of-seismic-source-heterogeneity-and-local-site-effects-on-intensity-attenuation-
This repository contains the data and code associated with the research paper: "The influence of seismic source heterogeneity and local site effects on intensity attenuation during Chilean megathrust earthquakes" submitted to Soil Dynamics and Earthquake Engeneering

1. Data
   
1.1 Main Calibration & Validation Dataset

daños_historicos_final_dataset.csv

        Description: This is the comprehensive master dataset used to calibrate and validate the intensity attenuation models.

        Content: It contains the compiled MSK-64 intensity observations for all seven studied earthquakes (1730, 1751, 1835, 1906, 1985, 2010, and 2015) along with the calculated distance metrics (e.g., Rhyp​, Rasp​, Raspmax​, Rasppond​) associated with each observation point.

1.2 High-Frequency (HF) & Site Response Datasets (Maule 2010)

daños_2010_dataset_[Model]_[Band].csv

        Description: A series of files containing MSK-64 intensities specifically for the 2010 Maule earthquake (Mw​ 8.8). These files are used to analyze the influence of High-Frequency (HF) radiation and local site effects.

        Content: Includes HF-based distance metrics and the soil fundamental frequency (f0​) for site response correction.

        Naming Convention: The filename suffix indicates the specific HF source model and frequency band used.

            Example: daños_2010_dataset_Palo_0.4_3_Hz.csv contains metrics calculated using the Palo et al. (2014) model within the 0.4–3 Hz frequency band.

1.3 Interseismic Locking Dataset

daños_2010_dataset_lock.csv

        Description: Dataset used to evaluate the predictive capability of interseismic coupling for future hazard scenarios.

        Content: Contains MSK-64 intensities for the 2010 Maule event with distance metrics derived exclusively from the Moreno et al. (2010) interseismic locking model, rather than coseismic slip distributions.

2. Codes
   
   2.1 Regression

      2.1.1 functions_MSK64.r
   
          Description: This script contains the R functions defining the governing equations for the Intensity Prediction Models tested in this study.
   
          Content: Each function returns the predicted intensity value along with a .grad attribute. This attribute contains the matrix of partial derivatives (gradient) with respect to the regression coefficients (c1​,c2​,…).
   
          This structure is specifically designed for the nlme (Non-Linear Mixed-Effects) R package, allowing the Gauss-Newton algorithm to converge faster and more accurately by avoiding finite-difference approximations of the derivative.
   
          Included Models:
      
                   Musson & Allen: Base form and variations without Magnitude (M) or with site effects (f0​).
               
                   Bakun & Wentworth: Including variations for calibration (with M) and validation (without M).
               
                   Atkinson: A more complex functional form including a magnitude-distance interaction term (M⋅logR).
      
      2.1.2 nls_attenuation_instrumental.r
   
          Description: This script performs the statistical calibration, validation, and specific sensitivity analysis of the Intensity Prediction Equations (IPEs). It utilizes Non-Linear Mixed-Effects (NLME) regression to account for inter-event and intra-event variability.
   
    
          Workflow:
   
                 Data Preprocessing:
   
                       Loads the master dataset (daños_historicos_final_dataset.csv).
               
                       Splits the data into a Calibration Set (Instrumental period: Year ≥ 1985) and a Validation Set (Historical period: Year < 1985).
   
                 Model Calibration (Instrumental Events):
   
                       Fits IPEs using the nlmer function from the lme4/nlme packages.
               
                       Tests multiple distance metrics: Rhyp​, Rasp​, Raspmax​, and Rasppond​.
               
                       Estimates fixed effects (coefficients c1​,c2​) and random effects (Event term).
               
                       Calculates goodness-of-fit statistics: AIC, τ (inter-event variability), ϕ (intra-event variability), and σ (total variability).
   
                       Output: coeff_metrics_MSK.csv (Model coefficients) and inter_event_residual_MSK.csv (δBe​).
   
                Model Validation (Historical Events):
   
                       Applies the fixed effects derived from the calibration step to the historical validation dataset.
               
                       Computes predicted intensities and residuals for events like 1730, 1835, and 1906.
               
                       Output: residuals_df_validation.csv.
   
                2010 Maule Earthquake & High-Frequency (HF) Analysis:
   
                       Loads specific datasets for the 2010 Maule event containing HF radiation metrics derived from Vera et al. (2024) and Palo et al. (2014) models across different frequency bands (0.5-2Hz, 1-4Hz, 2-8Hz, 0.4-3Hz).
               
                       Compares the performance of slip-based metrics vs. HF-based metrics (Rhf​, Rhfmax​) using linear regression (lm).
   
                Site Response Correction (f0​ Analysis):
   
                       Analyzes residuals from the 2010 event against the soil fundamental frequency (f0​).
               
                       Identifies systematic bias in specific frequency ranges.
               
                       Implements a correction term in the regression model for soft soils (f0​<1 Hz) and high-frequency sites (f0​≥5 Hz) using logarithmic dummy variables.
               
                       Output: coeff_metrics_2010_Palo_0.4_3Hz_f0.csv (Coefficients with site correction) and diagnostic plots.

   2.2 Plots

      2.2.1 plot_residuales_MSK.py

         Description: This script generates the residual analysis figures presented in the manuscript (Figures 3, 8, and S2) using PyGMT. It processes the output files from the regression analysis to visualize model performance, distance dependency, and site response bias.


         Workflow:

             Figure 3: Intra-event Residuals vs. Distance
         
                  Input: residuals_df_instrumental.csv
                  
                  Processing:
                  
                     Calculates statistical bins for residuals across logarithmic distance intervals (N=6 bins).
                  
                     Computes the mean and standard error (σ/N​) for each bin to visualize trends.
                  
                  Plotting: Generates a 4-panel figure comparing the distance metrics:
                  
                     (a) Rhyp​ vs (b) Raspmax​ vs (c) Rasp​ vs (d) Rasppond​.
                  
                     Visuals: Individual observations (red dots), binned means with error bars (blue triangles), and a zero-residual reference line.
                  
                  Output: Fig3.pdf
         
             Figure 8: Site Response Bias (f0​) for Raspmax​

                 Input: residuals_df_2010_Palo_0.4_3Hz.csv (2010 Maule event subset).
         
                 Processing:
         
                     Isolates the residuals of the best-performing metric (Raspmax​).
         
                     Plots the systematic bias functions derived from the regression for soft soils (f0​<1 Hz) and stiff sites (f0​>5 Hz).
         
                 Visuals: Log-scale x-axis (Frequency) vs. Residuals.
         
                 Output: Fig8.pdf

             Figure S2: Site Response Bias for High-Frequency Metrics

                 Input: coeff_metrics_2010_Palo_0.4_3Hz_f0.csv and residuals.
         
                 Processing:
         
                     Iterates through the High-Frequency (HF) metrics: Rhfmax​, Rhf​, and Rhfpond​.
         
                     Uses the regression coefficients (c3​,c4​) to plot the specific bias trend lines for each metric.
         
                 Plotting: Generates a 3-panel vertical figure (Supplementary Material).
         
                 Output: FigS2.pdf
   
   2.2.2 plot_observations.py

         Description: This script generates Figure 2 of the manuscript, illustrating the distribution of MSK-64 intensity observations as a function of distance. It visually compares the attenuation trends between the Calibration Dataset (Instrumental events) and the Validation Dataset (Historical events) across four different distance metrics.

         Workflow:
   
           Loads the master dataset (daños_historicos_final_dataset.csv).
      
           Splitting: Separates data into:
   
               Calibration Set: Events with Year >= 1985 (1985, 2010, 2015).
   
               Validation Set: Events with Year < 1985 (1730, 1751, 1835, 1906).

          Visualization (PyGMT):
   
              Creates a 2x2 multi-panel figure with a logarithmic X-axis (Distance) and linear Y-axis (Intensity).
      
              Panels:
      
                  (a) Hypocentral Distance (Rhyp​)
      
                  (b) Distance to Nearest Asperity (Rasp​)
      
                  (c) Distance to Max Slip Asperity (Raspmax​)
      
                  (d) Weighted Asperity Distance (Rasppond​)
      
              Symbology:
      
                  Red Circles: Calibration data (Instrumental).
      
                  Blue Triangles: Validation data (Historical).
      
              Formatting: Includes shared axes, specific annotations for "Distance [km]" and "MSK-64 Intensity", and a legend in the first panel.
   
            Output: Fig2.pdf

   2.2.3 plot_predictions.py

         Description: This script generates the comprehensive attenuation plots and spatial analysis maps for the manuscript (Figures 4, 5, 6, and 7). It visualizes the predictive performance of the calibrated models against observed data for instrumental and historical earthquakes, and performs specific spatial analyses for the 2010 Maule earthquake regarding interseismic locking and high-frequency (HF) radiation.

         Workflow:
      
          Calibration & Validation Attenuation Curves (Figures 4 & 5):
      
              Process: Iterates through all calibration (e.g., 2010, 2015) and validation (e.g., 1835, 1906) events.
      
              Plotting: Generates 4-panel figures comparing observations vs. predictions for metrics Rhyp​, Raspmax​, Rasp​, and Rasppond​.
      
              Features:
      
                  Mean prediction curves (solid black) and ±σ confidence intervals (gray polygons).
      
                  Statistical annotation: RMSE for calibration events; ubRMSE and Bias for validation events.
      
              Outputs: pred_[YEAR] (e.g., pred_2010 for Fig4.pdf and pred_1835 for Fig5.pdf).
      
          Interseismic Locking Analysis - Maule 2010 (Figure 6):
      
              Left Panel: Comparison of attenuation models (Raspmax​, Rasp​, Rasppond​) derived a priori from the Moreno et al. (2010) locking model.
      
              Right Panel (Map): A map of Central Chile visualizing:
      
                  Interseismic locking degree (Heatmap).
      
                  Comparison between Preseismic locking peaks (Green triangles) and Coseismic slip peaks (Blue triangles).
      
                  Megathrust segmentation (M1, M2, M3) according to segmentation model of Molina et al. (2021).
      
              Output: 2010_lock_pred, Fig6.pdf.
      
          High-Frequency (HF) Radiation Analysis - Maule 2010 (Figure 7):
      
              Left Panel: Comparison of the best slip-based metric (Raspmax​) against HF-based metrics (Rhf​, Rhfpond​) using the Palo et al. (2014) model (0.4–3 Hz band). Includes AIC and σ statistics.
      
              Right Panel (Map): A map visualizing the spatial distribution of source parameters (i.e. Slip peak, HF radiators and Hypocenter):
      
                  Normalized HF energy radiators (Red/Yellow circles).
      
                  Coseismic slip contours (Black lines) and peaks (Blue triangles).
      
                  Distribution of MSK-64 intensities (Green triangles).
      
              Output: 2010_hf_palo_0.4_3Hz_pred, Fig7.pdf.

   2.2.4 plot_rupture_lengths.py

         Description: This script generates Figure 1 of the manuscript, providing a comprehensive map of the study area in Central Chile. It visualizes the spatial distribution of the MSK-64 intensity dataset, the rupture extents of historical (validation dataset) and instrumental (calibration dataset) megathrust earthquakes (Mw​>8.0), and the tectonic segmentation of the Central Chile subduction zone according to Molina et. al. (2021).

         Workflow:
      
              Basemap: Defines a Mercator projection (M8c) covering the region [-81°, -69°] Longitude and [-43°, -27°] Latitude.
      
              Rupture Zones:
      
                  Manually plots the approximate rupture lengths of 12 major earthquakes spanning from 1730 to 2015 using colored lines.
      
                  Color Coding:
      
                      Red: Calibration events (Instrumental: 2015, 2010, 1985).
      
                      Blue: Validation events (Historical: 1730, 1751, 1835, 1906).
      
                      Black: Other significant historical events (e.g., 1822, 1880, 1928, 1943, 1971).
      
                  Labels are dynamically positioned based on event year to avoid overlap.
      
              Segmentation: Overlays the megathrust segmentation model (VP1-VP3, M1-M3) defined by Molina et al. (2021) using vector lines and text annotations.

   
            Output: Fig1.pdf

   2.2.5 plot_KDE_2010.py

         Description: This script generates Figure S1, which illustrates the methodology used to define the "Characteristic Asperity" model for the 2010 Maule earthquake (Mw​ 8.8). It visualizes the spatial convergence of multiple finite-fault models using a Kernel Density Estimation (KDE) approach to identify the most probable location of high-slip zones.

         Workflow:
      
          Inputs:    
                  KDE Grid: Loads the pre-calculated density grid (kde_2010.grd) derived from the centroids of available coseismic slip models.
      
                  All peaks : Centroids from individual slip models (peaks_2010.csv).
      
                  Characteristic peaks: Centroids of the characteristic asperity derived from KDE maxima (peaks_2010_kde.csv).
   
         Output: FigS1.pdf


   License The code in this repository is licensed under the MIT License. The datasets located in the Data/ folder are licensed under a Creative Commons Attribution 4.0 International License (CC-BY 4.0).

   
         
