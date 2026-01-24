# The-influence-of-seismic-source-heterogeneity-and-local-site-effects-on-intensity-attenuation-
This repository contains the data and code associated with the research paper: "The influence of seismic source heterogeneity and local site effects on intensity attenuation during Chilean megathrust earthquakes".

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

2. Code
