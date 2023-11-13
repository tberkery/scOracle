library(arrow)
library(stats)
library(Metrics)
library(tidyverse)
library(caret)

data = read_parquet('de_train.parquet')
training_data = data %>% filter(row_number() %% 5 != 0)

colnames(training_data)

use_pca = TRUE
use_cv = TRUE

if (use_pca) {
  numeric_data = training_data[, sapply(training_data, is.numeric)]
  scaled_data = scale(numeric_data)
  pca_result = prcomp(scaled_data, center = TRUE, scale = TRUE)
  summary(pca_result)
  plot(pca_result)
  # biplot(pca_result) # works but takes a long time to display... commenting for speed
  data = as.data.frame(pca_result$x) %>%
    cbind(numeric_data$A1BG) %>%
    rename(A1BG = 493) # rename last column to A1BG instead of numeric_data$A1BG
  data_sub = data %>%
    select(1:25, A1BG)
  
  # fresh glm with cross val to predict expression
  formula = A1BG ~ .
  ctrl = trainControl(method = "cv", number = 5)
  splitIndex = createDataPartition(y = data_sub$A1BG, p = 0.7, list = FALSE)
  train_data = data_sub[splitIndex, ]
  test_data = data_sub[-splitIndex, ]
  model = train(formula, data = train_data, method = "glm", trControl = ctrl)
  
  print(model)
  predictions = predict(model, newdata = test_data)
  #confusion_matrix = confusionMatrix(as.vector(predictions), as.vector(test_data$A1BG))
  #print(confusion_matrix)
  var_imp_by_pc = model$modelInfo$varImp(model)
  pc_num = 1:25
  var_imp = as.vector(var_imp_by_pc)
  plot_df = data.frame(pc_num, var_imp)
  ggplot(plot_df, aes(x = pc_num, y = var_imp)) +
    geom_point()
}

if (use_cv == FALSE) {
  
  glm = lm(formula = A1BG ~ cell_type + sm_name + control, data = training_data)
  summary(glm)
  
  train_test_data = data %>% filter(row_number() %% 5 == 0)
  predict_gene_exp = predict(glm, train_test_data)
  rmse(train_test_data$A1BG, predict_gene_exp)
  
  test_data = read_csv('id_map.csv')
  test_data$control = FALSE
  test_data$lfc = predict(glm, test_data)
}





