library(tidyverse)
library(arrow)
library(stats)
library(Metrics)
library(tidyverse)
library(caret)
library(tidymodels)
library(finetune)
library(vip)
library(butcher)
#library(bundle)

data = read_parquet('de_train.parquet')
genes = colnames(data)[6:length(colnames(data))]
num_genes = length(genes)

data_encoded = model.matrix(~ cell_type + sm_name + sm_lincs_id - 1, data) # one-hot encoding

data_encoded_oos = data_encoded %>%
  sample_frac(0.2)

num_runs = 2

data_to_decode = NULL

genes_options = genes_copy

models = vector(mode = "list", length = num_runs)

for (i in 1:num_runs) {
  print(paste0("Iteration #", i))
  genes_copy = genes_options
  
  first_gene_index = sample(1:length(genes_copy), 1)
  first_gene = genes_copy[[first_gene_index]]
  genes_copy = genes_copy[-first_gene_index]
  
  second_gene_index = sample(1:length(genes_copy), 1)
  second_gene = genes_copy[[second_gene_index]]
  genes_copy = genes_copy[-second_gene_index]
  
  third_gene_index = sample(1:length(genes_copy), 1)
  third_gene = genes_copy[[third_gene_index]]
  genes_copy = genes_copy[-third_gene_index]
  
  remaining_genes = genes_copy
  genes_options = genes_copy
  
  genes_of_interest_df = data %>%
    select(all_of(c(first_gene, second_gene, third_gene))) %>%
    rename(gene_1 = !!sym(first_gene),
           gene_2 = !!sym(second_gene),
           gene_3 = !!sym(third_gene)) %>%
    mutate(product = !!sym(first_gene) * !!sym(second_gene) * !!sym(third_gene))
  
  true_values = as.vector(genes_of_interest_df$product)
  
  train_df = data.frame(data_encoded)
  train_df$product = true_values
  
  # FIT XGBOOST MODEL HERE
  
  set.seed(123)
  
  split_data = train_df %>%
    initial_split()
  
  XGB_train = training(split_data)
  XGB_test = testing(split_data)
  
  set.seed(234)
  model_folds = vfold_cv(XGB_train, v = 5)
  
  XGB_rec =
    recipe(product ~ .,
           data = XGB_train
    ) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
  
  prep(XGB_rec)
  
  xgb_spec =
    boost_tree(
      trees = tune(),
      min_n = tune(),
      mtry = tune(),
      learn_rate = 0.01
    ) %>%
    set_engine("xgboost") %>%
    set_mode("regression")
  
  xgb_wf <- workflow(XGB_rec, xgb_spec)
  
  doParallel::registerDoParallel()
  
  set.seed(345)
  xgb_rs <- tune_race_anova(
    xgb_wf,
    resamples = model_folds,
    grid = 10,
    #metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )
  
  xgb_last = xgb_wf %>%
    tune::finalize_workflow(select_best(xgb_rs)) %>%
    tune::last_fit(split_data)
  
  xgb_last %>% saveRDS(paste0("./models/first_gene_", second_gene, "_", third_gene, "_xgb_last.rds"))
  
  xgb_metrics = xgb_last %>%
    workflowsets::collect_metrics()
  
  tree_specs = tune::select_best(xgb_rs)
  
  fitted_wf = workflowsets::extract_workflow(xgb_last) %>%
    butcher::butcher()
  
  fitted_wf2 = bundle::bundle(fitted_wf)
  
  fitted_wf %>% saveRDS(paste0("./models/first_gene_", second_gene, "_", third_gene, "_butcher_workflow.rds"))
  fitted_wf2 %>% saveRDS(paste0("./models/first_gene_", second_gene, "_", third_gene, "_bundle_workflow.rds"))
  
  preds = stats::predict(fitted_wf, train_df) %>%
    bind_cols(train_df)
  
  preds2 = collect_predictions(xgb_last)
  
  ggplot(preds, aes(.pred)) + geom_density()
  
  b = extract_workflow(xgb_last) %>%
    extract_fit_parsnip() %>%
    vip(geom = "point", num_features = 15)
  
  # work with preds
  product = preds
  
  add_df = NULL
  for (selected_gene in c(first_gene, second_gene, third_gene)) {
    new_addition_df = data.frame(product)
    expression = as.vector(data[[selected_gene]])
    new_addition_df = new_addition_df %>%
      mutate(gene_1 = first_gene,
             gene_2 = second_gene,
             gene_3 = third_gene) %>%
      mutate(decode_gene = selected_gene,
             decode_label = expression)
    add_df = rbind(add_df, new_addition_df)
  }
  data_to_decode = rbind(data_to_decode,
                         add_df)
  
  local_df = data_encoded %>%
    cbind(genes_of_interest_df)
  
}

# NOW FIT A DECODER XGBOOST

set.seed(123)

split_data = data_to_decode %>%
  initial_split()

XGB_train = training(split_data)
XGB_test = testing(split_data)

set.seed(234)
model_folds = vfold_cv(XGB_train, v = 5)

XGB_rec =
  recipe(decode_label ~ .,
         data = XGB_train
  ) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

prep(XGB_rec)

xgb_spec =
  boost_tree(
    trees = tune(),
    min_n = tune(),
    mtry = tune(),
    learn_rate = 0.01
  ) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

xgb_wf <- workflow(XGB_rec, xgb_spec)

doParallel::registerDoParallel()

set.seed(345)
xgb_rs <- tune_race_anova(
  xgb_wf,
  resamples = model_folds,
  grid = 10,
  #metrics = metric_set(mn_log_loss),
  control = control_race(verbose_elim = TRUE)
)

xgb_last = xgb_wf %>%
  tune::finalize_workflow(select_best(xgb_rs)) %>%
  tune::last_fit(split_data)

xgb_last %>% saveRDS(paste0("./models/decoder_xgb_last.rds"))

xgb_metrics = xgb_last %>%
  workflowsets::collect_metrics()

tree_specs = tune::select_best(xgb_rs)

fitted_wf = workflowsets::extract_workflow(xgb_last) %>%
  butcher::butcher()

fitted_wf2 = bundle::bundle(fitted_wf)

fitted_wf %>% saveRDS(paste0("./models/decoder_butcher_workflow.rds"))
fitted_wf2 %>% saveRDS(paste0("./models/decoder_bundle_workflow.rds"))

preds = stats::predict(fitted_wf, data_to_decode) %>%
  bind_cols(data_to_decode)

preds2 = collect_predictions(xgb_last)

ggplot(preds, aes(.pred)) + geom_density()

b = extract_workflow(xgb_last) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point", num_features = 15)

# NOW APPLY ENCODER AND DECODER XGBOOSTS TO OOS DATA

new_df = data_encoded_oos

calculate_mrsme <- function(obs, pred) {
  # Check if lengths match
  if (length(obs) != length(pred)) {
    stop("Lengths of observed and predicted vectors do not match.")
  }
  
  # Convert vectors to matrices with one column each
  obs_matrix <- matrix(obs, ncol = 1)
  pred_matrix <- matrix(pred, ncol = 1)
  
  # Calculate squared relative mean error for each element
  squared_relative_error <- (obs_matrix - pred_matrix)^2 / obs_matrix^2
  
  # Calculate mean squared relative error
  mrsme <- mean(squared_relative_error, na.rm = TRUE)
  
  return(mrsme)
}
