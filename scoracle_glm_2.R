setwd('~/Downloads/open-problems-single-cell-perturbations/')
library(caret)
library(arrow)
library(stats)
library(Metrics)
library(tidyverse)#

data = read_parquet('de_train.parquet')
genes = colnames(data)[6:length(colnames(data))]

data_b_m = data[data$cell_type == 'B cells' | data$cell_type == 'Myeloid cells',]
data_t_n = data[data$cell_type != 'B cells' & data$cell_type != 'Myeloid cells',]
splitIndexBM = sample.int(34, 10)
splitIndexTN = sample.int(580, 420)
train_data_BM = data_b_m[splitIndexBM, ]
train_data_TN = data_t_n[splitIndexTN, ]
test_data_BM = data_b_m[-splitIndexBM, ]
test_data_TN = data_t_n[-splitIndexTN, ]
train_data = rbind(train_data_BM, train_data_TN)
test_data = rbind(test_data_BM, test_data_TN)


rmse = map_dbl(genes, function(gene) {
  g = train_data[[gene]]
  ind = !is.nan(g) & !is.infinite(g) & !is.na(g)
  g = g[ind]
  cell_type = train_data$cell_type[ind]
  sm_name = train_data$sm_name[ind]
  control = train_data$control[ind]
  glm = lm(formula = g ~ cell_type + sm_name + control)
  
  pred = predict(glm, test_data)
  rmse(test_data[[gene]], pred)
})

plot_tb = as.data.frame(tibble(rmse = rmse))
ggplot(data = plot_tb) +
  geom_boxplot(aes(x = rmse))

mean(rmse)
sd(rmse)
