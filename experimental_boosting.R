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
library(bundle)

data = read_parquet('de_train.parquet')
genes = colnames(data)[6:length(colnames(data))]
num_genes = length(genes)

data_encoded = model.matrix(~ cell_type + sm_name + sm_lincs_id + control - 1, data) # one-hot encoding

num_runs = 50

data_to_decode = NULL

for (i in 1:num_runs) {
  genes_copy = genes
  
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
  
  genes_of_interest_df = data %>%
    select(all_of(c(first_gene, second_gene, third_gene))) %>%
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
  
  xgb_metrics = xgb_last %>%
    workflowsets::collect_metrics()
  
  tree_specs = tune::select_best(xgb_rs)
  
  fitted_wf = workflowsets::extract_workflow(xgb_last) %>%
    butcher::butcher()
  
  #fitted_wf2 = bundle::bundle(fitted_wf)
  
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