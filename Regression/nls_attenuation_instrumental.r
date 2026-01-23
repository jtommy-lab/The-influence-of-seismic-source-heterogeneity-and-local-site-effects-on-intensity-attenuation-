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


path_functions = '/home/jtommy/Escritorio/Respaldo/functions/functions_MSK64.r'
source(path_functions)

data_MSK64 = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_historicos_final_dataset.csv',check.names = FALSE)
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

write.csv(coeff_df,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/coeff_metrics_MSK.csv')

regression_df_T$residuals_Rhyp = residuals(fit_Rhyp)
regression_df_T$residuals_Rasp = residuals(fit_Rasp)
regression_df_T$residuals_Rasp_max = residuals(fit_Rasp_max)
regression_df_T$residuals_Rasp_pond = residuals(fit_Rasp_pond)

write.csv(regression_df_T,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/residuals_df_instrumental.csv')

dBe_df = data.frame(Rhyp = ranef_Rhyp$EVENT$c1,Rasp_max = ranef_Rasp_max$EVENT$c1,Rasp = ranef_Rasp$EVENT$c1,Rasp_pond = ranef_Rasp_pond$EVENT$c1)
rownames(dBe_df) = rownames(ranef_Rasp_pond$EVENT)


write.csv(dBe_df,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/inter_event_residual_MSK.csv')

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


write.csv(regression_df_T,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/residuals_df_validation.csv')


####### Generamos predicciones con Rhf para 2010 #######

### Vera 0.5-2Hz ###

data_felipe_Rasp = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe.csv',check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe_HF_vera.csv',check.names = FALSE)

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

write.csv(coeff_df,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/coeff_metrics_2010_vera.csv')


### Palo 1-4Hz ###

data_felipe_Rasp = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe.csv',check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe_HF_Palo_1_4_Hz.csv',check.names = FALSE)

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

write.csv(coeff_df,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/coeff_metrics_2010_Palo_1_4Hz.csv')


### Palo 2-8Hz ###

data_felipe_Rasp = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe.csv',check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe_HF_Palo_2_8_Hz.csv',check.names = FALSE)

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

write.csv(coeff_df,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/coeff_metrics_2010_Palo_2_8Hz.csv')


### Palo 0.4-3Hz ###

data_felipe_Rasp = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe.csv',check.names = FALSE)
data_felipe_Rasp = subset(data_felipe_Rasp,data_felipe_Rasp$Period == 0)
data_felipe_Rhf = read.csv('/home/jtommy/Escritorio/Respaldo/Paper2_v2/dataset/daños_2010_dataset_Felipe_HF_Palo_0.4_3_Hz.csv',check.names = FALSE)

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

write.csv(coeff_df,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/coeff_metrics_2010_Palo_0.4_3Hz.csv')


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

write.csv(coeff_df,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/coeff_metrics_2010_Palo_0.4_3Hz_f0.csv')


regression_df_T$residuals_Rasp_max_f0 = residuals(fit_Rasp_max_f0)
regression_df_T$residuals_Rhf_max_f0 = residuals(fit_Rhf_max_f0)
regression_df_T$residuals_Rhf_f0 = residuals(fit_Rhf_f0)
regression_df_T$residuals_Rhf_pond_f0 = residuals(fit_Rhf_pond_f0)

write.csv(regression_df_T,'/home/jtommy/Escritorio/Respaldo/Paper2_v2/Stats_results/residuals_df_2010_Palo_0.4_3Hz.csv')

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












####### Con datos de Felipe y Fernandez ########

df1 = data.frame(MSK64 = as.numeric(data_MSK64_felipe$MSK), Rhyp = data_MSK64_felipe$"Rhyp [km]", Rp = data_MSK64_felipe$"Rp [km]",  M = 8.8,
                            Rasp = data_MSK64_felipe$"Rasp [km]",Rasp_max = data_MSK64_felipe$"Rasp max [km]",Rasp_pond = data_MSK64_felipe$"Rasp pond slip [km]",
                            EVENT = "2010",f0 = data_MSK64_felipe$"Frecuencia")
df2 = data.frame(MSK64 = as.numeric(data_MSK64_fernandez$MSK64), Rhyp = data_MSK64_fernandez$"Rhyp [km]", Rp = data_MSK64_fernandez$"Rp [km]",  M = 8.3,
                            Rasp = data_MSK64_fernandez$"Rasp [km]",Rasp_max = data_MSK64_fernandez$"Rasp max [km]",Rasp_pond = data_MSK64_fernandez$"Rasp pond slip [km]",
                            EVENT = "2015",f0 = 1/data_MSK64_fernandez$"Tp")
df2 = subset(df2, df2$f0 > 0)  

regression_df_T = df1#rbind(df1,df2)

rownames(regression_df_T) <- 1:nrow(regression_df_T)

regression_df_T$log10_f0<- ifelse(
  regression_df_T$f0 <= 1,
  log10(regression_df_T$f0),
  0
)


###### Musson y Allen sin M ######

#start_values_ini = c(c1 = 9.94, c2 = -2)

ggplot()+
geom_point(data = regression_df_T, aes(x = Rasp_pond, y = MSK64), color = 'blue')+
scale_x_log10()+
theme_minimal()

fit_Rhyp_nof0 <- lm(MSK64 ~ log10(Rhyp) , data = regression_df_T)
#fit_Rhyp_f0 <- lm(MSK64 ~ log(Rhyp)+ Rhyp + log10_f0, data = regression_df_T)
fit_Rasp_nof0 = lm(MSK64 ~ log10(Rasp) , data = regression_df_T)
fit_Rasp_max_nof0 = lm(MSK64 ~ log10(Rasp_max), data = regression_df_T)
#fit_Rasp_max_f0 = lm(MSK64 ~ log10(Rasp_max)+ Rasp_max + log10_f0, data = regression_df_T)
#fit_Rasp_max = lm(MSK64 ~ log10(Rasp_max)+ Rasp_max + log10(f0), data = regression_df_T)
fit_Rasp_pond_nof0 = lm(MSK64 ~ log10(Rasp_pond) , data = regression_df_T)

fit_Rhyp_f0 <- lm(MSK64 ~ log10(Rhyp) + log10_f0, data = regression_df_T)
#fit_Rhyp_f0 <- lm(MSK64 ~ log(Rhyp)+ Rhyp + log10_f0, data = regression_df_T)
fit_Rasp_f0 = lm(MSK64 ~ log10(Rasp) + log10_f0, data = regression_df_T)
fit_Rasp_max_f0 = lm(MSK64 ~ log10(Rasp_max) + log10_f0, data = regression_df_T)
#fit_Rasp_max_f0 = lm(MSK64 ~ log10(Rasp_max)+ Rasp_max + log10_f0, data = regression_df_T)
#fit_Rasp_max = lm(MSK64 ~ log10(Rasp_max)+ Rasp_max + log10(f0), data = regression_df_T)
fit_Rasp_pond_f0 = lm(MSK64 ~ log10(Rasp_pond) + log10_f0, data = regression_df_T)










start_values_Rhyp = c(c1 = coef(fit_Rhyp_ini)[['(Intercept)']], c2 = coef(fit_Rhyp_ini)[['log10(Rhyp)']],D = coef(fit_Rhyp_ini)[['log10_f0']])
start_values_Rasp = c(c1 = coef(fit_Rasp_ini)[['(Intercept)']], c2 = coef(fit_Rasp_ini)[['log10(Rasp)']],D = coef(fit_Rasp_ini)[['log10_f0']])
start_values_Rasp_max = c(c1 = coef(fit_Rasp_max_ini)[['(Intercept)']], c2 = coef(fit_Rasp_max_ini)[['log10(Rasp_max)']],D = coef(fit_Rasp_max_ini)[['log10_f0']])
start_values_Rasp_pond = c(c1 = coef(fit_Rasp_pond_ini)[['(Intercept)']], c2 = coef(fit_Rasp_pond_ini)[['log10(Rasp_pond)']],D = coef(fit_Rasp_pond_ini)[['log10_f0']])

## Modelo Musson y Allen ##
fit_Rhyp_Musson <- nlmer(MSK64 ~ Musson_y_Allen_noM_f0(c1,c2,D,f0,Rhyp) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                        start = start_values_Rhyp[names(start_values_Rhyp) %in% c('c1','c2','D')],nAGQ = 1L)
fit_Rasp_Musson <- nlmer(MSK64 ~ Musson_y_Allen_noM_f0(c1,c2,D,f0,Rasp) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                        start = start_values_Rasp[names(start_values_Rasp) %in% c('c1','c2','D')],nAGQ = 1L)
fit_Rasp_max_Musson <- nlmer(MSK64 ~ Musson_y_Allen_noM_f0(c1,c2,D,f0,Rasp_max) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                            start = start_values_Rasp_max[names(start_values_Rasp_max) %in% c('c1','c2','D')],nAGQ = 1L)
fit_Rasp_pond_Musson <- nlmer(MSK64 ~ Musson_y_Allen_noM_f0(c1,c2,D,f0,Rasp_pond) ~ c1 + c2 + (c1+0|EVENT),data = regression_df_T,
                            start = start_values_Rasp_pond[names(start_values_Rasp_pond) %in% c('c1','c2','D')],nAGQ = 1L)

## Residuales ##
residuals_Rhyp_Musson = resid(fit_Rhyp_Musson)
residuals_Rasp_Musson = resid(fit_Rasp_Musson)
residuals_Rasp_max_Musson = resid(fit_Rasp_max_Musson)
residuals_Rasp_pond_Musson = resid(fit_Rasp_pond_Musson)
regression_df_T$residuals_Rhyp_Musson = residuals_Rhyp_Musson
regression_df_T$residuals_Rasp_Musson = residuals_Rasp_Musson
regression_df_T$residuals_Rasp_max_Musson = residuals_Rasp_max_Musson
regression_df_T$residuals_Rasp_pond_Musson = residuals_Rasp_pond_Musson

## plot residuals ##
ggplot()+
  geom_point(data = regression_df_T, aes(x = f0, y = residuals_Rhyp_Musson), color = 'blue')+
  geom_hline(yintercept = 0, linetype='dashed', color='red')+
  scale_x_log10()+
  theme_minimal()

ggplot()+
  geom_point(data = regression_df_T, aes(x = f0, y = residuals_Rasp_Musson), color = 'blue')+
  geom_hline(yintercept = 0, linetype='dashed', color='red')+
  scale_x_log10()+
  theme_minimal()

ggplot()+
  geom_point(data = regression_df_T, aes(x = f0, y = residuals_Rasp_max_Musson), color = 'blue')+
  geom_hline(yintercept = 0, linetype='dashed', color='red')+
  scale_x_log10()+
  theme_minimal()   

ggplot()+
  geom_point(data = regression_df_T, aes(x = Rasp_max, y = residuals_Rasp_max_Musson), color = 'blue')+
  geom_hline(yintercept = 0, linetype='dashed', color='red')+
  scale_x_log10()+
  theme_minimal()   

ggplot()+
  geom_point(data = regression_df_T, aes(x = f0, y = residuals_Rasp_pond_Musson), color = 'blue')+
  geom_hline(yintercept = 0, linetype='dashed', color='red')+
  scale_x_log10()+
  theme_minimal()


# start_values_ini = c(A = 9.94, B = -3.71,C = -0.0013)
# fitparam = nls.control(maxiter = 1000, tol = 1e-05, minFactor = 1/2048)

# ### Initial regression to obtain start values ###
# fit_Rhyp_ini = nls(MSK64 ~ Bakun_noM(A,B,C,Rhyp),data = regression_df_T, start = start_values_ini, control=fitparam,)
# start_values_Rhyp = c(A = coef(fit_Rhyp_ini)[['A']], B = coef(fit_Rhyp_ini)[['B']],C = coef(fit_Rhyp_ini)[['C']])

# fit_Rasp_ini = nls(MSK64 ~ Bakun_noM(A,B,C,Rasp),data = regression_df_T, start = start_values_ini, control=fitparam,)
# start_values_Rasp = c(A = coef(fit_Rasp_ini)[['A']], B = coef(fit_Rasp_ini)[['B']],C = coef(fit_Rasp_ini)[['C']])

# fit_Rasp_max_ini = nls(MSK64 ~ Bakun_noM(A,B,C,Rasp_max),data = regression_df_T, start = start_values_ini, control=fitparam,)
# start_values_Rasp_max = c(A = coef(fit_Rasp_max_ini)[['A']], B = coef(fit_Rasp_max_ini)[['B']],C = coef(fit_Rasp_max_ini)[['C']])

# fit_Rasp_pond_ini = nls(MSK64 ~ Bakun_noM(A,B,C,Rasp_pond),data = regression_df_T, start = start_values_ini, control=fitparam,)
# start_values_Rasp_pond = c(A = coef(fit_Rasp_pond_ini)[['A']], B = coef(fit_Rasp_pond_ini)[['B']],C = coef(fit_Rasp_pond_ini)[['C']])

    

## Modelo Bakun ##


# fit_Rhyp <- nlmer(MSK64 ~ Bakun_noM(A,B,C,Rhyp) ~ A + B + C + (A+0|EVENT),data = regression_df_T,
#                         start = start_values_Rhyp[names(start_values_Rhyp) %in% c('A','B','C')],nAGQ = 1L)
# fit_Rasp <- nlmer(MSK64 ~ Bakun_noM(A,B,C,Rasp) ~ A + B + C + (A+0|EVENT),data = regression_df_T,
#                         start = start_values_Rasp[names(start_values_Rasp) %in% c('A','B','C')],nAGQ = 1L)
# fit_Rasp_max <- nlmer(MSK64 ~ Bakun_noM(A,B,C,Rasp_max) ~ A + B + C + (A+0|EVENT),data = regression_df_T,
#                             start = start_values_Rasp_max[names(start_values_Rasp_max) %in% c('A','B','C')],nAGQ = 1L)
# fit_Rasp_pond <- nlmer(MSK64 ~ Bakun_noM(A,B,C,Rasp_pond) ~ A + B + C + (A+0|EVENT),data = regression_df_T,
#                             start = start_values_Rasp_pond[names(start_values_Rasp_pond) %in% c('A','B','C')],nAGQ = 1L)



fit_Rhyp_ini <- lm(MSK64 ~ log10(Rhyp)+ Rhyp + log10_f0, data = regression_df_T)
#fit_Rhyp_f0 <- lm(MSK64 ~ log(Rhyp)+ Rhyp + log10_f0, data = regression_df_T)
fit_Rasp_ini = lm(MSK64 ~ log10(Rasp)+ Rasp + log10_f0, data = regression_df_T)
fit_Rasp_max_ini = lm(MSK64 ~ log10(Rasp_max)+ Rasp_max + log10_f0, data = regression_df_T)
#fit_Rasp_max_f0 = lm(MSK64 ~ log10(Rasp_max)+ Rasp_max + log10_f0, data = regression_df_T)
#fit_Rasp_max = lm(MSK64 ~ log10(Rasp_max)+ Rasp_max + log10(f0), data = regression_df_T)
fit_Rasp_pond_ini = lm(MSK64 ~ log10(Rasp_pond)+ Rasp_pond + log10_f0, data = regression_df_T)

start_values_Rhyp = c(A = coef(fit_Rhyp_ini)[['(Intercept)']], B = coef(fit_Rhyp_ini)[['log10(Rhyp)']],C = coef(fit_Rhyp_ini)[['Rhyp']],D = coef(fit_Rhyp_ini)[['log10_f0']])

start_values_Rasp = c(A = coef(fit_Rasp_ini)[['(Intercept)']], B = coef(fit_Rasp_ini)[['log10(Rasp)']],C = coef(fit_Rasp_ini)[['Rasp']],D = coef(fit_Rasp_ini)[['log10_f0']])

start_values_Rasp_max = c(A = coef(fit_Rasp_max_ini)[['(Intercept)']], B = coef(fit_Rasp_max_ini)[['log10(Rasp_max)']],C = coef(fit_Rasp_max_ini)[['Rasp_max']],D = coef(fit_Rasp_max_ini)[['log10_f0']])
start_values_Rasp_pond = c(A = coef(fit_Rasp_pond_ini)[['(Intercept)']], B = coef(fit_Rasp_pond_ini)[['log10(Rasp_pond)']],C = coef(fit_Rasp_pond_ini)[['Rasp_pond']],D = coef(fit_Rasp_pond_ini)[['log10_f0']])
fit_Rhyp = nlmer(MSK64 ~ Bakun_noM_f0(A,B,C,D,Rhyp,f0) ~ A + B + C + D + (A+0|EVENT),data = regression_df_T,
                        start = start_values_Rhyp[names(start_values_Rhyp) %in% c('A','B','C','D')],nAGQ = 1L)
fit_Rasp = nlmer(MSK64 ~ Bakun_noM_f0(A,B,C,D,Rasp,f0) ~ A + B + C + D + (A+0|EVENT),data = regression_df_T,
                        start = start_values_Rasp[names(start_values_Rasp) %in% c('A','B','C','D')],nAGQ = 1L)
fit_Rasp_max = nlmer(MSK64 ~ Bakun_noM_f0(A,B,C,D,Rasp_max,f0) ~ A + B + C + D + (A+0|EVENT),data = regression_df_T,
                            start = start_values_Rasp_max[names(start_values_Rasp_max) %in% c('A','B','C','D')],nAGQ = 1L)
fit_Rasp_pond = nlmer(MSK64 ~ Bakun_noM_f0(A,B,C,D,Rasp_pond,f0) ~ A + B + C + D + (A+0|EVENT),data = regression_df_T,
                            start = start_values_Rasp_pond[names(start_values_Rasp_pond) %in% c('A','B','C','D')],nAGQ = 1L)

residuals_Rasp_max = residuals(fit_Rasp_max)
residuals_Rasp = residuals(fit_Rasp)
residuals_Rhyp = residuals(fit_Rhyp)
residuals_Rasp_pond = residuals(fit_Rasp_pond)

regression_df_T$residuals_Rasp_max = residuals_Rasp_max
regression_df_T$residuals_Rasp = residuals_Rasp
regression_df_T$residuals_Rhyp = residuals_Rhyp
regression_df_T$residuals_Rasp_pond = residuals_Rasp_pond

#fit_residuals_f0 = lm(residuals_Rasp_max ~ log10(f0) + I(log10(f0)^2), data = regression_df_T)   

ggplot()+
    geom_point(data=regression_df_T, aes(x=f0, y=residuals_Rasp_max), color='blue', alpha=0.5)+
    labs(title=paste0('Residuals Rasp max vs f0 - Period = ',T), x='f0 (Hz)', y='Residuals Rasp max')+
    geom_hline(yintercept = 0, linetype='dashed', color='red')+
    ylim(-2,2)+
    scale_x_log10()+
    theme_minimal() 

ggplot()+
    geom_point(data=regression_df_T, aes(x=f0, y=residuals_Rasp), color='blue', alpha=0.5)+
    labs(title=paste0('Residuals Rasp vs f0 - Period = ',T), x='f0 (Hz)', y='Residuals Rasp')+
    geom_hline(yintercept = 0, linetype='dashed', color='red')+
    scale_x_log10()+
    theme_minimal() 

ggplot()+
    geom_point(data=regression_df_T, aes(x=f0, y=residuals_Rhyp), color='blue', alpha=0.5)+
    labs(title=paste0('Residuals Rhyp vs f0 - Period = ',T), x='f0 (Hz)', y='Residuals Rhyp')+
    geom_hline(yintercept = 0, linetype='dashed', color='red')+
    ylim(-2,2)+
    scale_x_log10()+
    theme_minimal()

ggplot()+
    geom_point(data=regression_df_T, aes(x=f0, y=residuals_Rasp_pond), color='blue', alpha=0.5)+
    labs(title=paste0('Residuals Rasp pond vs f0 - Period = ',T), x='f0 (Hz)', y='Residuals Rasp pond')+
    geom_hline(yintercept = 0, linetype='dashed', color='red')+
    ylim(-2,2)+
    scale_x_log10()+
    theme_minimal()  
