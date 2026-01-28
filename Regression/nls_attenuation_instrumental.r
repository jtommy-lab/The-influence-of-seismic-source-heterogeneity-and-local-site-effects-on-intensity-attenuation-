library(nlme)
library(lme4)
library(ggplot2)
library(dplyr)
library(data.table)
library(readxl)
library(minpack.lm)
library(caret)
library(tidyr)
library(patchwork)
library(here)


path_functions = here("Regression", "functions_MSK64.r")
source(path_functions)


output_dir = here("Regression", "Stats_results")


if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


data_MSK64 = read.csv(here("Data", "daños_historicos_final_dataset.csv"), check.names = FALSE)

data_MSK64 = subset(data_MSK64, data_MSK64$Period == 0)
data_MSK64_instrumental = subset(data_MSK64,data_MSK64$Year >= 1985)
data_MSK64_validation = subset(data_MSK64,data_MSK64$Year < 1985)


regression_df_T = data.frame(MSK64 = as.numeric(data_MSK64_instrumental$Intensity), Rhyp = data_MSK64_instrumental$"Rhyp [km]", Rp = data_MSK64_instrumental$"Rp [km]",  M = data_MSK64_instrumental$Magnitude,
                            Rasp = data_MSK64_instrumental$"Rasp [km]",Rasp_max = data_MSK64_instrumental$"Rasp max [km]",Rasp_pond = data_MSK64_instrumental$"Rasp pond slip [km]",
                            EVENT = as.character(data_MSK64_instrumental$Year))
regression_df_T = na.omit(regression_df_T)

n_obs_event <- regression_df_T %>%
  count(EVENT, name = "n_obs")

print(n_obs_event)


### Initial regression to obtain start values ###
fit_Rhyp_ini = lm(MSK64 ~ log10(Rhyp), data = regression_df_T)
start_values_Rhyp = c(c1 = coef(fit_Rhyp_ini)[['(Intercept)']], c2 = coef(fit_Rhyp_ini)[['log10(Rhyp)']])

fit_Rasp_max_ini = lm(MSK64 ~ log10(Rasp_max) , data = regression_df_T)
start_values_Rasp_max = c(c1 = coef(fit_Rasp_max_ini)[['(Intercept)']], c2 = coef(fit_Rasp_max_ini)[['log10(Rasp_max)']])

fit_Rasp_ini = lm(MSK64 ~ log10(Rasp), data = regression_df_T)
start_values_Rasp = c(c1 = coef(fit_Rasp_ini)[['(Intercept)']], c2 = coef(fit_Rasp_ini)[['log10(Rasp)']])


fit_Rasp_pond_ini = lm(MSK64 ~ log10(Rasp_pond), data = regression_df_T)
start_values_Rasp_pond = c(c1 = coef(fit_Rasp_pond_ini)[['(Intercept)']], c2 = coef(fit_Rasp_pond_ini)[['log10(Rasp_pond)']])





fit_Rhyp <- nlmer(MSK64 ~ Musson_y_Allen_noM(c1,c2,Rhyp) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                        start = start_values_Rhyp[names(start_values_Rhyp) %in% c('c1','c2')],nAGQ = 1L)
AIC_Rhyp = AIC(fit_Rhyp)
tau_Rhyp = attr(VarCorr(fit_Rhyp)$EVENT, "stddev")[['c1']]
phi_Rhyp = attr(VarCorr(fit_Rhyp),"sc")
sigma_Rhyp = sqrt(tau_Rhyp^2 + phi_Rhyp^2)
ranef_Rhyp = ranef(fit_Rhyp)
fixef_Rhyp = fixef(fit_Rhyp)
se_Rhyp = sqrt(diag(vcov(fit_Rhyp)))
     

fit_Rasp_max <- nlmer(MSK64 ~ Musson_y_Allen_noM(c1,c2,Rasp_max) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                        start = start_values_Rasp_max[names(start_values_Rasp_max) %in% c('c1','c2')],nAGQ = 1L)
AIC_Rasp_max = AIC(fit_Rasp_max)
tau_Rasp_max = attr(VarCorr(fit_Rasp_max)$EVENT, "stddev")[['c1']]
phi_Rasp_max = attr(VarCorr(fit_Rasp_max),"sc")
sigma_Rasp_max = sqrt(tau_Rasp_max^2 + phi_Rasp_max^2)
ranef_Rasp_max = ranef(fit_Rasp_max)
fixef_Rasp_max = fixef(fit_Rasp_max)
se_Rasp_max = sqrt(diag(vcov(fit_Rasp_max)))

fit_Rasp <- nlmer(MSK64 ~ Musson_y_Allen_noM(c1,c2,Rasp) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                        start = start_values_Rasp[names(start_values_Rasp) %in% c('c1','c2')],nAGQ = 1L)
AIC_Rasp = AIC(fit_Rasp)
tau_Rasp = attr(VarCorr(fit_Rasp)$EVENT, "stddev")[['c1']]
phi_Rasp = attr(VarCorr(fit_Rasp),"sc")
sigma_Rasp = sqrt(tau_Rasp^2 + phi_Rasp^2)
ranef_Rasp = ranef(fit_Rasp)
fixef_Rasp = fixef(fit_Rasp)
se_Rasp = sqrt(diag(vcov(fit_Rasp)))


fit_Rasp_pond <- nlmer(MSK64 ~ Musson_y_Allen_noM(c1,c2,Rasp_pond) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                            start = start_values_Rasp_pond[names(start_values_Rasp_pond) %in% c('c1','c2')],nAGQ = 1L)
AIC_Rasp_pond = AIC(fit_Rasp_pond)
tau_Rasp_pond = attr(VarCorr(fit_Rasp_pond)$EVENT, "stddev")[['c1']]
phi_Rasp_pond = attr(VarCorr(fit_Rasp_pond),"sc")
sigma_Rasp_pond = sqrt(tau_Rasp_pond^2 + phi_Rasp_pond^2)
ranef_Rasp_pond = ranef(fit_Rasp_pond)
fixef_Rasp_pond = fixef(fit_Rasp_pond)
se_Rasp_pond = sqrt(diag(vcov(fit_Rasp_pond)))

coeff_df = data.frame(Rhyp = c(fixef_Rhyp[['c1']],fixef_Rhyp[['c2']],se_Rhyp[['c1']],se_Rhyp[['c2']],sigma_Rhyp),
                      Rasp_max = c(fixef_Rasp_max[['c1']],fixef_Rasp_max[['c2']],se_Rasp_max[['c1']],se_Rasp_max[['c2']],sigma_Rasp_max),
                      Rasp = c(fixef_Rasp[['c1']],fixef_Rasp[['c2']],se_Rasp[['c1']],se_Rasp[['c2']],sigma_Rasp),
                      Rasp_pond = c(fixef_Rasp_pond[['c1']],fixef_Rasp_pond[['c2']],se_Rasp_pond[['c1']],se_Rasp_pond[['c2']],sigma_Rasp_pond))

rownames(coeff_df) = c('c1','c2','se_c1','se_c2','sigma')

write.csv(coeff_df, file.path(output_dir, "coeff_metrics_MSK.csv"))

regression_df_T$residuals_Rhyp = residuals(fit_Rhyp)
regression_df_T$residuals_Rasp = residuals(fit_Rasp)
regression_df_T$residuals_Rasp_max = residuals(fit_Rasp_max)
regression_df_T$residuals_Rasp_pond = residuals(fit_Rasp_pond)

write.csv(regression_df_T,file.path(output_dir,"residuals_df_instrumental.csv"))

dBe_df = data.frame(Rhyp = ranef_Rhyp$EVENT$c1,Rasp_max = ranef_Rasp_max$EVENT$c1,Rasp = ranef_Rasp$EVENT$c1,Rasp_pond = ranef_Rasp_pond$EVENT$c1)
rownames(dBe_df) = rownames(ranef_Rasp_pond$EVENT)


write.csv(dBe_df,file.path(output_dir,"inter_event_residual_MSK.csv"))

predict(fit_Rhyp)


#### Predictions for validation dataset ####

regression_df_T = data.frame(MSK64 = as.numeric(data_MSK64_validation$Intensity), Rhyp = data_MSK64_validation$"Rhyp [km]", Rp = data_MSK64_validation$"Rp [km]",  M = data_MSK64_validation$Magnitude,
                            Rasp = data_MSK64_validation$"Rasp [km]",Rasp_max = data_MSK64_validation$"Rasp max [km]",Rasp_pond = data_MSK64_validation$"Rasp pond slip [km]",
                            EVENT = as.character(data_MSK64_validation$Year))
regression_df_T = na.omit(regression_df_T)


pred_Rhyp = as.numeric(Bakun_noM(fixef(fit_Rhyp)['A'],fixef(fit_Rhyp)['B'],fixef(fit_Rhyp)['C'],
                                regression_df_T$Rhyp))
regression_df_T$pred_Rhyp = pred_Rhyp


pred_Rasp_max = as.numeric(Bakun_noM(fixef(fit_Rasp_max)['A'],fixef(fit_Rasp_max)['B'],fixef(fit_Rasp_max)['C'],
                                regression_df_T$Rasp_max))
regression_df_T$pred_Rasp_max = pred_Rasp_max

pred_Rasp = as.numeric(Musson_y_Allen_noM(fixef(fit_Rasp)['c1'],fixef(fit_Rasp)['c2'],
                                regression_df_T$Rasp))
regression_df_T$pred_Rasp = pred_Rasp


pred_Rasp_pond = as.numeric(Musson_y_Allen_noM(fixef(fit_Rasp_pond)['c1'],fixef(fit_Rasp_pond)['c2'],
                                regression_df_T$Rasp_pond))
regression_df_T$pred_Rasp_pond = pred_Rasp_pond


write.csv(regression_df_T,file.path(output_dir,"residuals_df_validation.csv"))


####### Generamos predicciones con Rhf para 2010 #######

### Vera 0.5-2Hz ###

data_felipe_Rasp = read.csv(here("Data","daños_2010_dataset_Felipe.csv"),check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv(here("Data","daños_2010_dataset_Felipe_HF_vera.csv"),check.names = FALSE)

regression_df_T = data.frame(MSK64 = as.numeric(data_felipe_Rasp$MSK), Rhyp = data_felipe_Rasp$"Rhyp [km]",Rasp = data_felipe_Rasp$"Rasp [km]",
                            Rasp_max = data_felipe_Rasp$"Rasp max [km]",Rasp_pond = data_felipe_Rasp$"Rasp pond slip [km]",Rhf = data_felipe_Rhf$Rhf,
                            Rhf_max = data_felipe_Rhf$Rhf_max,Rhf_pond = data_felipe_Rhf$Rhf_pond)

### Initial regression to obtain start values ###

fit_Rasp_max = lm(MSK64 ~ log(Rasp_max) , data = regression_df_T)


fit_Rhf_max = lm(MSK64 ~ log(Rhf_max) , data = regression_df_T)


fit_Rhf = lm(MSK64 ~ log(Rhf) , data = regression_df_T)


fit_Rhf_pond = lm(MSK64 ~ log(Rhf_pond) , data = regression_df_T)

AIC_Rasp_max = AIC(fit_Rasp_max)
se_Rasp_max = sqrt(diag(vcov(fit_Rasp_max)))
sigma_Rasp_max = summary(fit_Rasp_max)$sigma

AIC_Rhf_max = AIC(fit_Rhf_max)
se_Rhf_max = sqrt(diag(vcov(fit_Rhf_max)))
sigma_Rhf_max = summary(fit_Rhf_max)$sigma

AIC_Rhf = AIC(fit_Rhf)
se_Rhf = sqrt(diag(vcov(fit_Rhf)))
sigma_Rhf = summary(fit_Rhf)$sigma

AIC_Rhf_pond = AIC(fit_Rhf_pond)
se_Rhf_pond = sqrt(diag(vcov(fit_Rhf_pond)))
sigma_Rhf_pond = summary(fit_Rhf_pond)$sigma


coeff_df = data.frame(Rasp_max = c(coef(fit_Rasp_max)[["(Intercept)"]],coef(fit_Rasp_max)[["log(Rasp_max)"]],se_Rasp_max[["(Intercept)"]],se_Rasp_max[["log(Rasp_max)"]],sigma_Rasp_max,AIC_Rasp_max),
                      Rhf_max = c(coef(fit_Rhf_max)[["(Intercept)"]],coef(fit_Rhf_max)[["log(Rhf_max)"]],se_Rhf_max[["(Intercept)"]],se_Rhf_max[["log(Rhf_max)"]],sigma_Rhf_max,AIC_Rhf_max),
                      Rhf = c(coef(fit_Rhf)[["(Intercept)"]],coef(fit_Rhf)[["log(Rhf)"]],se_Rhf[["(Intercept)"]],se_Rhf[["log(Rhf)"]],sigma_Rhf,AIC_Rhf),
                      Rhf_pond = c(coef(fit_Rhf_pond)[["(Intercept)"]],coef(fit_Rhf_pond)[["log(Rhf_pond)"]],se_Rhf_pond[["(Intercept)"]],se_Rhf_pond[["log(Rhf_pond)"]],sigma_Rhf_pond,AIC_Rhf_pond))

rownames(coeff_df) = c('c1','c2','se_c1','se_c2','sigma','AIC')

write.csv(coeff_df,file.path(output_dir,"coeff_metrics_2010_vera.csv"))


### Palo 1-4Hz ###

data_felipe_Rasp = read.csv(here("Data","daños_2010_dataset_Felipe.csv"),check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv(here("Data","daños_2010_dataset_Felipe_HF_Palo_1_4_Hz.csv"),check.names = FALSE)

regression_df_T = data.frame(MSK64 = as.numeric(data_felipe_Rasp$MSK), Rhyp = data_felipe_Rasp$"Rhyp [km]",Rasp = data_felipe_Rasp$"Rasp [km]",
                            Rasp_max = data_felipe_Rasp$"Rasp max [km]",Rasp_pond = data_felipe_Rasp$"Rasp pond slip [km]",Rhf = data_felipe_Rhf$Rhf,
                            Rhf_max = data_felipe_Rhf$Rhf_max,Rhf_pond = data_felipe_Rhf$Rhf_pond)

### Initial regression to obtain start values ###

fit_Rasp_max = lm(MSK64 ~ log(Rasp_max) , data = regression_df_T)


fit_Rhf_max = lm(MSK64 ~ log(Rhf_max) , data = regression_df_T)


fit_Rhf = lm(MSK64 ~ log(Rhf) , data = regression_df_T)


fit_Rhf_pond = lm(MSK64 ~ log(Rhf_pond) , data = regression_df_T)


AIC_Rasp_max = AIC(fit_Rasp_max)
se_Rasp_max = sqrt(diag(vcov(fit_Rasp_max)))
sigma_Rasp_max = summary(fit_Rasp_max)$sigma

AIC_Rhf_max = AIC(fit_Rhf_max)
se_Rhf_max = sqrt(diag(vcov(fit_Rhf_max)))
sigma_Rhf_max = summary(fit_Rhf_max)$sigma

AIC_Rhf = AIC(fit_Rhf)
se_Rhf = sqrt(diag(vcov(fit_Rhf)))
sigma_Rhf = summary(fit_Rhf)$sigma

AIC_Rhf_pond = AIC(fit_Rhf_pond)
se_Rhf_pond = sqrt(diag(vcov(fit_Rhf_pond)))
sigma_Rhf_pond = summary(fit_Rhf_pond)$sigma


coeff_df = data.frame(Rasp_max = c(coef(fit_Rasp_max)[["(Intercept)"]],coef(fit_Rasp_max)[["log(Rasp_max)"]],se_Rasp_max[["(Intercept)"]],se_Rasp_max[["log(Rasp_max)"]],sigma_Rasp_max,AIC_Rasp_max),
                      Rhf_max = c(coef(fit_Rhf_max)[["(Intercept)"]],coef(fit_Rhf_max)[["log(Rhf_max)"]],se_Rhf_max[["(Intercept)"]],se_Rhf_max[["log(Rhf_max)"]],sigma_Rhf_max,AIC_Rhf_max),
                      Rhf = c(coef(fit_Rhf)[["(Intercept)"]],coef(fit_Rhf)[["log(Rhf)"]],se_Rhf[["(Intercept)"]],se_Rhf[["log(Rhf)"]],sigma_Rhf,AIC_Rhf),
                      Rhf_pond = c(coef(fit_Rhf_pond)[["(Intercept)"]],coef(fit_Rhf_pond)[["log(Rhf_pond)"]],se_Rhf_pond[["(Intercept)"]],se_Rhf_pond[["log(Rhf_pond)"]],sigma_Rhf_pond,AIC_Rhf_pond))

rownames(coeff_df) = c('c1','c2','se_c1','se_c2','sigma','AIC')

write.csv(coeff_df,file.path(output_dir,"coeff_metrics_2010_Palo_1_4Hz.csv"))


### Palo 2-8Hz ###

data_felipe_Rasp = read.csv(here("Data","daños_2010_dataset_Felipe.csv"),check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv(here("Data","daños_2010_dataset_Felipe_HF_Palo_2_8_Hz.csv"),check.names = FALSE)

regression_df_T = data.frame(MSK64 = as.numeric(data_felipe_Rasp$MSK), Rhyp = data_felipe_Rasp$"Rhyp [km]",Rasp = data_felipe_Rasp$"Rasp [km]",
                            Rasp_max = data_felipe_Rasp$"Rasp max [km]",Rasp_pond = data_felipe_Rasp$"Rasp pond slip [km]",Rhf = data_felipe_Rhf$Rhf,
                            Rhf_max = data_felipe_Rhf$Rhf_max,Rhf_pond = data_felipe_Rhf$Rhf_pond)

### Initial regression to obtain start values ###

fit_Rasp_max = lm(MSK64 ~ log(Rasp_max) , data = regression_df_T)


fit_Rhf_max = lm(MSK64 ~ log(Rhf_max) , data = regression_df_T)


fit_Rhf = lm(MSK64 ~ log(Rhf) , data = regression_df_T)


fit_Rhf_pond = lm(MSK64 ~ log(Rhf_pond) , data = regression_df_T)


AIC_Rasp_max = AIC(fit_Rasp_max)
se_Rasp_max = sqrt(diag(vcov(fit_Rasp_max)))
sigma_Rasp_max = summary(fit_Rasp_max)$sigma

AIC_Rhf_max = AIC(fit_Rhf_max)
se_Rhf_max = sqrt(diag(vcov(fit_Rhf_max)))
sigma_Rhf_max = summary(fit_Rhf_max)$sigma

AIC_Rhf = AIC(fit_Rhf)
se_Rhf = sqrt(diag(vcov(fit_Rhf)))
sigma_Rhf = summary(fit_Rhf)$sigma

AIC_Rhf_pond = AIC(fit_Rhf_pond)
se_Rhf_pond = sqrt(diag(vcov(fit_Rhf_pond)))
sigma_Rhf_pond = summary(fit_Rhf_pond)$sigma


coeff_df = data.frame(Rasp_max = c(coef(fit_Rasp_max)[["(Intercept)"]],coef(fit_Rasp_max)[["log(Rasp_max)"]],se_Rasp_max[["(Intercept)"]],se_Rasp_max[["log(Rasp_max)"]],sigma_Rasp_max,AIC_Rasp_max),
                      Rhf_max = c(coef(fit_Rhf_max)[["(Intercept)"]],coef(fit_Rhf_max)[["log(Rhf_max)"]],se_Rhf_max[["(Intercept)"]],se_Rhf_max[["log(Rhf_max)"]],sigma_Rhf_max,AIC_Rhf_max),
                      Rhf = c(coef(fit_Rhf)[["(Intercept)"]],coef(fit_Rhf)[["log(Rhf)"]],se_Rhf[["(Intercept)"]],se_Rhf[["log(Rhf)"]],sigma_Rhf,AIC_Rhf),
                      Rhf_pond = c(coef(fit_Rhf_pond)[["(Intercept)"]],coef(fit_Rhf_pond)[["log(Rhf_pond)"]],se_Rhf_pond[["(Intercept)"]],se_Rhf_pond[["log(Rhf_pond)"]],sigma_Rhf_pond,AIC_Rhf_pond))

rownames(coeff_df) = c('c1','c2','se_c1','se_c2','sigma','AIC')

write.csv(coeff_df,file.path(output_dir,"coeff_metrics_2010_Palo_2_8Hz.csv"))


### Palo 0.4-3Hz ###

data_felipe_Rasp = read.csv(here("Data","daños_2010_dataset_Felipe.csv"),check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv(here("Data","daños_2010_dataset_Felipe_HF_Palo_0.4_3_Hz.csv"),check.names = FALSE)

regression_df_T = data.frame(MSK64 = as.numeric(data_felipe_Rasp$MSK), Rhyp = data_felipe_Rasp$"Rhyp [km]",Rasp = data_felipe_Rasp$"Rasp [km]",
                            Rasp_max = data_felipe_Rasp$"Rasp max [km]",Rasp_pond = data_felipe_Rasp$"Rasp pond slip [km]",Rhf = data_felipe_Rhf$Rhf,
                            Rhf_max = data_felipe_Rhf$Rhf_max,Rhf_pond = data_felipe_Rhf$Rhf_pond,f0 = data_felipe_Rhf$Frecuencia)

### Initial regression to obtain start values ###

fit_Rasp_max = lm(MSK64 ~ log(Rasp_max), data = regression_df_T)


fit_Rhf_max = lm(MSK64 ~ log(Rhf_max) , data = regression_df_T)


fit_Rhf = lm(MSK64 ~ log(Rhf) , data = regression_df_T)


fit_Rhf_pond = lm(MSK64 ~ log(Rhf_pond) , data = regression_df_T)

AIC_Rasp_max = AIC(fit_Rasp_max)
se_Rasp_max = sqrt(diag(vcov(fit_Rasp_max)))
sigma_Rasp_max = summary(fit_Rasp_max)$sigma

AIC_Rhf_max = AIC(fit_Rhf_max)
se_Rhf_max = sqrt(diag(vcov(fit_Rhf_max)))
sigma_Rhf_max = summary(fit_Rhf_max)$sigma

AIC_Rhf = AIC(fit_Rhf)
se_Rhf = sqrt(diag(vcov(fit_Rhf)))
sigma_Rhf = summary(fit_Rhf)$sigma

AIC_Rhf_pond = AIC(fit_Rhf_pond)
se_Rhf_pond = sqrt(diag(vcov(fit_Rhf_pond)))
sigma_Rhf_pond = summary(fit_Rhf_pond)$sigma


coeff_df = data.frame(Rasp_max = c(coef(fit_Rasp_max)[["(Intercept)"]],coef(fit_Rasp_max)[["log(Rasp_max)"]],se_Rasp_max[["(Intercept)"]],se_Rasp_max[["log(Rasp_max)"]],sigma_Rasp_max,AIC_Rasp_max),
                      Rhf_max = c(coef(fit_Rhf_max)[["(Intercept)"]],coef(fit_Rhf_max)[["log(Rhf_max)"]],se_Rhf_max[["(Intercept)"]],se_Rhf_max[["log(Rhf_max)"]],sigma_Rhf_max,AIC_Rhf_max),
                      Rhf = c(coef(fit_Rhf)[["(Intercept)"]],coef(fit_Rhf)[["log(Rhf)"]],se_Rhf[["(Intercept)"]],se_Rhf[["log(Rhf)"]],sigma_Rhf,AIC_Rhf),
                      Rhf_pond = c(coef(fit_Rhf_pond)[["(Intercept)"]],coef(fit_Rhf_pond)[["log(Rhf_pond)"]],se_Rhf_pond[["(Intercept)"]],se_Rhf_pond[["log(Rhf_pond)"]],sigma_Rhf_pond,AIC_Rhf_pond))

rownames(coeff_df) = c('c1','c2','se_c1','se_c2','sigma','AIC')

write.csv(coeff_df,file.path(output_dir,"coeff_metrics_2010_Palo_0.4_3Hz.csv"))


regression_df_T$residuals_Rasp_max = residuals(fit_Rasp_max)
regression_df_T$residuals_Rhf_max = residuals(fit_Rhf_max)
regression_df_T$residuals_Rhf = residuals(fit_Rhf)
regression_df_T$residuals_Rhf_pond = residuals(fit_Rhf_pond)




p1 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rasp_max))+
scale_x_log10()

p2 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rhf_max))+
scale_x_log10()

p3 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rhf))+
scale_x_log10()

p4 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rhf_pond))+
scale_x_log10()

p = p1/p2/p3/p4


### Encontramos un sesgo en el modelo con respecto a f0, por lo tanto, lo agregamos al modelo ###

regression_df_T$logf0_1<- ifelse(
  regression_df_T$f0 < 1,
  log10(regression_df_T$f0),
  0
)

regression_df_T$logf0_5<- ifelse(
  regression_df_T$f0 >=5 ,
  log10(regression_df_T$f0),
  0
)
subset(regression_df_T,regression_df_T$f0 >=5)

regression_df_T

fit_Rasp_max_f0 = lm(MSK64 ~ log(Rasp_max) + logf0_1 + logf0_5, data = regression_df_T)


fit_Rhf_max_f0 = lm(MSK64 ~ log(Rhf_max) + logf0_1 + logf0_5, data = regression_df_T)


fit_Rhf_f0 = lm(MSK64 ~ log(Rhf) + logf0_1 + logf0_5, data = regression_df_T)


fit_Rhf_pond_f0 = lm(MSK64 ~ log(Rhf_pond) + logf0_1 + logf0_5, data = regression_df_T)


AIC_Rasp_max = AIC(fit_Rasp_max_f0)
se_Rasp_max = sqrt(diag(vcov(fit_Rasp_max_f0)))
sigma_Rasp_max = summary(fit_Rasp_max_f0)$sigma

AIC_Rhf_max = AIC(fit_Rhf_max_f0)
se_Rhf_max = sqrt(diag(vcov(fit_Rhf_max_f0)))
sigma_Rhf_max = summary(fit_Rhf_max_f0)$sigma

AIC_Rhf = AIC(fit_Rhf_f0)
se_Rhf = sqrt(diag(vcov(fit_Rhf_f0)))
sigma_Rhf = summary(fit_Rhf_f0)$sigma

AIC_Rhf_pond = AIC(fit_Rhf_pond_f0)
se_Rhf_pond = sqrt(diag(vcov(fit_Rhf_pond_f0)))
sigma_Rhf_pond = summary(fit_Rhf_pond_f0)$sigma


coeff_df = data.frame(Rasp_max = c(coef(fit_Rasp_max_f0)[["(Intercept)"]],coef(fit_Rasp_max_f0)[["log(Rasp_max)"]],coef(fit_Rasp_max_f0)[["logf0_1"]],coef(fit_Rasp_max_f0)[["logf0_5"]],
                                  se_Rasp_max[["(Intercept)"]],se_Rasp_max[["log(Rasp_max)"]],se_Rasp_max[["logf0_1"]],se_Rasp_max[["logf0_5"]],sigma_Rasp_max,AIC_Rasp_max),
                      Rhf_max = c(coef(fit_Rhf_max_f0)[["(Intercept)"]],coef(fit_Rhf_max_f0)[["log(Rhf_max)"]],coef(fit_Rhf_max_f0)[["logf0_1"]],coef(fit_Rhf_max_f0)[["logf0_5"]],
                                  se_Rhf_max[["(Intercept)"]],se_Rhf_max[["log(Rhf_max)"]],se_Rhf_max[["logf0_1"]],se_Rhf_max[["logf0_5"]],sigma_Rhf_max,AIC_Rhf_max),
                      Rhf = c(coef(fit_Rhf_f0)[["(Intercept)"]],coef(fit_Rhf_f0)[["log(Rhf)"]],coef(fit_Rhf_f0)[["logf0_1"]],coef(fit_Rhf_f0)[["logf0_5"]],
                                  se_Rhf[["(Intercept)"]],se_Rhf[["log(Rhf)"]],se_Rhf[["logf0_1"]],se_Rhf[["logf0_5"]],sigma_Rhf,AIC_Rhf),
                      Rhf_pond = c(coef(fit_Rhf_pond_f0)[["(Intercept)"]],coef(fit_Rhf_pond_f0)[["log(Rhf_pond)"]],coef(fit_Rhf_pond_f0)[["logf0_1"]],coef(fit_Rhf_pond_f0)[["logf0_5"]],
                                  se_Rhf_pond[["(Intercept)"]],se_Rhf_pond[["log(Rhf_pond)"]],se_Rhf_pond[["logf0_1"]],se_Rhf_pond[["logf0_5"]],sigma_Rhf_pond,AIC_Rhf_pond))

rownames(coeff_df) = c('c1','c2','c3','c4','se_c1','se_c2','se_c3','se_c4','sigma','AIC')

write.csv(coeff_df,file.path(output_dir,"coeff_metrics_2010_Palo_0.4_3Hz_f0.csv"))


regression_df_T$residuals_Rasp_max_f0 = residuals(fit_Rasp_max_f0)
regression_df_T$residuals_Rhf_max_f0 = residuals(fit_Rhf_max_f0)
regression_df_T$residuals_Rhf_f0 = residuals(fit_Rhf_f0)
regression_df_T$residuals_Rhf_pond_f0 = residuals(fit_Rhf_pond_f0)

write.csv(regression_df_T,file.path(output_dir,"residuals_df_2010_Palo_0.4_3Hz.csv"))

p1 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rasp_max_f0))+
scale_x_log10()

p2 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rhf_max_f0))+
scale_x_log10()

p3 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rhf_f0))+
scale_x_log10()

p4 = ggplot()+
geom_point(data = regression_df_T,aes(x = f0, y = residuals_Rhf_pond_f0))+
scale_x_log10()

p = p1/p2/p3/p4
