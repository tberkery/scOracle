library(arrow)
library(stats)
library(Metrics)
library(tidyvers)

data = read_parquet('de_train.parquet')
training_data = data %>% filter(row_number() %% 5 != 0)

colnames(training_data)

use_pca = FALSE
if (use_pca) {
  numeric_data <- training_data[, sapply(training_data, is.numeric)]
  scaled_data <- scale(numeric_data)
  pca_result <- prcomp(scaled_data, center = TRUE, scale = TRUE)
  summary(pca_result)
  plot(pca_result)
  biplot(pca_result)
}

glm = lm(formula = A1BG ~ cell_type + sm_name + control, data = training_data)
summary(glm)

train_test_data = data %>% filter(row_number() %% 5 == 0)
predict_gene_exp = predict(glm, train_test_data)
rmse(train_test_data$A1BG, predict_gene_exp)

test_data = read_csv('id_map.csv')
test_data$control = FALSE
test_data$lfc = predict(glm, test_data)
Collapse


glm = lm(formula = A1BG ~ cell_type + sm_name + control, data = training_data)
summary(glm)

train_test_data = data %>% filter(row_number() %% 5 == 0)
predict_gene_exp = predict(glm, train_test_data)
rmse(train_test_data$A1BG, predict_gene_exp)

test_data = read_csv('id_map.csv')
test_data$control = FALSE
test_data$lfc = predict(glm, test_data)
Collapse











