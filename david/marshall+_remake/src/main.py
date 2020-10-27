# Imports
import gc
import matplotlib.pyplot as plt
import os
import statsmodels.stats.proportion
import sklearn.pipeline

# seaborn
import seaborn
seaborn.set()
seaborn.set_style("darkgrid")

# Project imports
from legacy_model import *

# Get raw data
raw_data = get_raw_scdb_data("../data/input/SCDB_Legacy_01_justiceCentered_Citation.csv")

# Get feature data
if os.path.exists("../data/output/feature_data.hdf.gz"):
    print("Loading from HDF5 cache")
    feature_df = pandas.read_hdf("../data/output/feature_data.hdf.gz", "root")
else:
    # Process
    feature_df = preprocess_raw_data(raw_data, include_direction=True)
    
    # Write out feature datas
    feature_df.to_hdf("../data/output/feature_data.hdf.gz", "root", complevel=6, complib="zlib")

# Downsample to float
feature_df = feature_df.astype(numpy.float16)

# Remove term
nonterm_features = [f for f in feature_df.columns if not f.startswith("term_")]
original_feature_df = feature_df.copy()
feature_df = original_feature_df.loc[:, nonterm_features].copy()
gc.collect()

# Output some diagnostics on features
print(raw_data.shape)
print(feature_df.shape)
assert(raw_data.shape[0] == feature_df.shape[0])

# Reset output file timestamp per run
file_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Reset seed per run
numpy.random.seed(0)

# Setup training time period
dummy_window = 10
min_training_years = 25
term_range = range(raw_data["term"].min() + min_training_years,
                   raw_data["term"].max() + 1)

# Setting growing random forest parameters
# Number of trees to grow per term
trees_per_term = 5

# Number of trees to begin with
initial_trees = min_training_years * trees_per_term

# Setup model
m = None
term_count = 0
feature_importance_df = pandas.DataFrame()

for term in term_range:
    # Diagnostic output
    print("Term: {0}".format(term))
    term_count += 1
    
    # Setup train and test periods
    train_index = (raw_data.loc[:, "term"] < term).values
    dummy_train_index = ((raw_data.loc[:, "term"] < term) & (raw_data.loc[:, "term"] >= (term-dummy_window))).values
    test_index = (raw_data.loc[:, "term"] == term).values
    if test_index.sum() == 0:
        continue
    
    # Setup train data
    feature_data_train = feature_df.loc[train_index, :]
    target_data_train = (raw_data.loc[train_index, "justice_outcome_disposition"]).astype(int)
    target_data_weights = target_data_train.value_counts() / target_data_train.shape[0]

    # Setup test data
    feature_data_test = feature_df.loc[test_index, :]
    target_data_test = (raw_data.loc[test_index, "justice_outcome_disposition"]).astype(int)
    
    # Check if the justice set has changed
    if set(raw_data.loc[raw_data.loc[:, "term"] == (term-1), "naturalCourt"].unique()) != \
        set(raw_data.loc[raw_data.loc[:, "term"] == (term), "naturalCourt"].unique()):
        # natural Court change; trigger forest fire
        print("Natural court change; rebuilding with {0} trees".format(initial_trees + (term_count * trees_per_term)))
        m = None

    # Build or grow a model depending on initial/reset condition
    if not m:
        # Grow an initial forest
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=initial_trees + (term_count * trees_per_term),
                                                     class_weight=target_data_weights.to_dict(),
                                                    warm_start=True,
                                                    n_jobs=-1)
        pipeline = sklearn.pipeline.Pipeline([("rf", rf)
                                             ])
        
        search_params = {"rf__min_samples_leaf": [1, 2]}
        m = sklearn.model_selection.GridSearchCV(pipeline, search_params, scoring='accuracy')
    else:
        # Grow the forest by increasing the number of trees (requires warm_start=True)
        m.set_params(estimator__rf__n_estimators=initial_trees + (term_count * trees_per_term))

    # Fit the forest model
    m.fit(feature_data_train,
          target_data_train)
    #print(m.get_params())
    
    # Record feature weights
    current_feature_importance_df = pandas.DataFrame(list(zip(feature_df.columns, m.best_estimator_.steps[-1][-1].feature_importances_)),
                                         columns=["feature", "importance"])
    current_feature_importance_df.loc[:, "term"] = term
    if feature_importance_df.shape[0] == 0:
        feature_importance_df = current_feature_importance_df.copy()
    else:
        feature_importance_df = feature_importance_df.append(current_feature_importance_df.copy())

    # Fit the "dummy" model
    d = sklearn.dummy.DummyClassifier(strategy="most_frequent")
    d.fit(feature_df.loc[dummy_train_index, :],
          (raw_data.loc[dummy_train_index, "justice_outcome_disposition"]).astype(int))
    
    # Perform forest predictions
    raw_data.loc[test_index, "rf_predicted"] = m.predict(feature_data_test)
    
    # Store scores per class
    scores = m.predict_proba(feature_data_test)
    raw_data.loc[test_index, "rf_predicted_score_other"] = scores[:, 0]
    raw_data.loc[test_index, "rf_predicted_score_affirm"] = scores[:, 1]
    raw_data.loc[test_index, "rf_predicted_score_reverse"] = scores[:, 2]
    
    # Store dummy predictions
    raw_data.loc[test_index, "dummy_predicted"] = d.predict(feature_data_test)
    
    #  Clear
    del feature_data_train
    del feature_data_test
    del target_data_train
    del target_data_test
    gc.collect()

    # Evaluation range
evaluation_index = raw_data.loc[:, "term"].isin(term_range)
target_actual = (raw_data.loc[evaluation_index, "justice_outcome_disposition"]).astype(int)
target_predicted = raw_data.loc[evaluation_index, "rf_predicted"].astype(int)
target_dummy = raw_data.loc[evaluation_index, "dummy_predicted"].astype(int)
raw_data.loc[:, "rf_correct"] = numpy.nan
raw_data.loc[:, "dummy_correct"] = numpy.nan
raw_data.loc[evaluation_index, "rf_correct"] = (target_actual == target_predicted).astype(float)
raw_data.loc[evaluation_index, "dummy_correct"] = (target_actual == target_dummy).astype(float)

# Setup reverse testing
reverse_target_actual = (raw_data.loc[evaluation_index, "justice_outcome_disposition"] > 0).astype(int)
reverse_target_predicted = (raw_data.loc[evaluation_index, "rf_predicted"] > 0).astype(int)
reverse_target_dummy = (raw_data.loc[evaluation_index, "dummy_predicted"] > 0).astype(int)
raw_data.loc[:, "rf_reverse_correct"] = numpy.nan
raw_data.loc[:, "dummy_reverse_correct"] = numpy.nan
raw_data.loc[evaluation_index, "rf_reverse_correct"] = (reverse_target_actual == reverse_target_predicted).astype(float)
raw_data.loc[evaluation_index, "dummy_reverse_correct"] = (reverse_target_actual == reverse_target_dummy).astype(float)

# Compare model
print("RF model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_predicted))
print(sklearn.metrics.confusion_matrix(target_actual, target_predicted))
print(sklearn.metrics.accuracy_score(target_actual, target_predicted))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_dummy))
print(sklearn.metrics.confusion_matrix(target_actual, target_dummy))
print(sklearn.metrics.accuracy_score(target_actual, target_dummy))
print("="*32)
print("")

# Compare model
print("RF model")
print("="*32)
print(sklearn.metrics.classification_report(reverse_target_actual, reverse_target_predicted))
print(sklearn.metrics.confusion_matrix(reverse_target_actual, reverse_target_predicted))
print(sklearn.metrics.accuracy_score(reverse_target_actual, reverse_target_predicted))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(sklearn.metrics.classification_report(reverse_target_actual, reverse_target_dummy))
print(sklearn.metrics.confusion_matrix(reverse_target_actual, reverse_target_dummy))
print(sklearn.metrics.accuracy_score(reverse_target_actual, reverse_target_dummy))
print("="*32)
print("")

# Setup time series
rf_correct_ts = raw_data.loc[evaluation_index, :].groupby("term")["rf_correct"].mean()
dummy_correct_ts = raw_data.loc[evaluation_index, :].groupby("term")["dummy_correct"].mean()
rf_reverse_correct_ts = raw_data.loc[evaluation_index, :].groupby("term")["rf_reverse_correct"].mean()
dummy_reverse_correct_ts = raw_data.loc[evaluation_index, :].groupby("term")["dummy_reverse_correct"].mean()


# Plot all accuracies
f = plt.figure(figsize=(16, 12))
plt.plot(rf_reverse_correct_ts.index, rf_reverse_correct_ts,
         marker='o', alpha=0.75)
plt.plot(dummy_reverse_correct_ts.index, dummy_reverse_correct_ts,
         marker='>', alpha=0.75)
plt.legend(('Random forest', 'Dummy'))

# Setup time series
rf_spread_ts = rf_reverse_correct_ts - dummy_reverse_correct_ts

# Plot all accuracies
f = plt.figure(figsize=(16, 12))
plt.bar(rf_spread_ts.index, rf_spread_ts,
        alpha=0.75)
plt.xlabel("Term")
plt.ylabel("Spread (%)")
plt.title("Spread over dummy model for justice accuracy")

# Output stats
print("t-test:")
print("Uncalibrated:")
print(scipy.stats.ttest_rel(rf_correct_ts.values,
                   dummy_correct_ts.values))

print("=" * 16)
print("ranksum-test:")
print("Uncalibrated:")
print(scipy.stats.ranksums(rf_correct_ts.values,
                   dummy_correct_ts.values))

print("=" * 16)
print("Binomial:")
print(statsmodels.stats.proportion.binom_test(raw_data.loc[evaluation_index, "rf_correct"].sum(),
                                              raw_data.loc[evaluation_index, "rf_correct"].shape[0],
                                              raw_data.loc[evaluation_index, "dummy_correct"].mean(),
                                              alternative="larger"))

# Output stats
print("t-test:")
print("Uncalibrated:")
print(scipy.stats.ttest_rel(rf_reverse_correct_ts.values,
                   dummy_reverse_correct_ts.values))

print("=" * 16)
print("ranksum-test:")
print("Uncalibrated:")
print(scipy.stats.ranksums(rf_reverse_correct_ts.values,
                   dummy_reverse_correct_ts.values))

print("=" * 16)
print("Binomial:")
print(statsmodels.stats.proportion.binom_test(raw_data.loc[evaluation_index, "rf_reverse_correct"].sum(),
                                              raw_data.loc[evaluation_index, "rf_reverse_correct"].shape[0],
                                              raw_data.loc[evaluation_index, "dummy_reverse_correct"].mean(),
                                              alternative="larger"))

# Feature importance
last_feature_importance_df = pandas.DataFrame(list(zip(feature_df.columns, m.best_estimator_.steps[-1][-1].feature_importances_)),
                                         columns=["feature", "importance"])
last_feature_importance_df.sort_values(["importance"], ascending=False).head(10)

# Get outcomes as reverse/not-reverse for real data
raw_data.loc[:, "justice_outcome_reverse"] = (raw_data.loc[:, "justice_outcome_disposition"] > 0).astype(int)
raw_data.loc[:, "case_outcome_reverse"] = (raw_data.loc[:, "case_outcome_disposition"] > 0).astype(int)

# Store reverse predictions
raw_data.loc[evaluation_index, "rf_predicted_reverse"] = (raw_data.loc[evaluation_index, "rf_predicted"] > 0).astype(int)
raw_data.loc[evaluation_index, "dummy_predicted_reverse"] = (raw_data.loc[evaluation_index, "dummy_predicted"] > 0).astype(int)

# Group by case
rf_predicted_case = (raw_data.loc[evaluation_index, :]\
    .groupby("docketId")["rf_predicted_reverse"].mean() >= 0.5).astype(int)

dummy_predicted_case = (raw_data.loc[evaluation_index, :]\
    .groupby("docketId")["dummy_predicted_reverse"].mean() >= 0.5).astype(int)

actual_case = (raw_data.loc[evaluation_index, :]\
    .groupby("docketId")["case_outcome_reverse"].mean() > 0).astype(int)

# Setup case dataframe
case_data = pandas.DataFrame(rf_predicted_case).join(dummy_predicted_case).join(actual_case)
case_data = case_data.join(raw_data.groupby("docketId")[["term", "naturalCourt"]].mean().astype(int))

# Setup correct columns
case_data.loc[:, "rf_correct_case"] = numpy.nan
case_data.loc[:, "dummy_correct_case"] = numpy.nan
case_data.loc[:, "rf_correct_case"] = (case_data.loc[:, "rf_predicted_reverse"] == case_data.loc[:, "case_outcome_reverse"])\
    .astype(int)
case_data.loc[:, "dummy_correct_case"] = (case_data.loc[:, "dummy_predicted_reverse"] == case_data.loc[:, "case_outcome_reverse"])\
    .astype(int)

# Join back onto raw data
case_data.loc[:, "docketId"] = case_data.index
raw_data = raw_data.join(case_data.loc[:, ["docketId", "rf_correct_case", "dummy_correct_case"]], on="docketId", rsuffix="_case")

# Output comparison
# Evaluation range
evaluation_index = case_data.loc[:, "term"].isin(term_range)
target_actual = case_data.loc[:, "case_outcome_reverse"]
target_predicted = case_data.loc[:, "rf_predicted_reverse"]
target_dummy = case_data.loc[:, "dummy_predicted_reverse"]

# Compare model
print("RF model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_predicted))
print(sklearn.metrics.confusion_matrix(target_actual, target_predicted))
print(sklearn.metrics.accuracy_score(target_actual, target_predicted))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_dummy))
print(sklearn.metrics.confusion_matrix(target_actual, target_dummy))
print(sklearn.metrics.accuracy_score(target_actual, target_dummy))
print("="*32)
print("")

# Output comparison
# Evaluation range
last_century = case_data["term"].drop_duplicates().sort_values().tail(100)
evaluation_index = case_data.loc[:, "term"].isin(last_century)
target_actual = case_data.loc[evaluation_index, "case_outcome_reverse"]
target_predicted = case_data.loc[evaluation_index, "rf_predicted_reverse"]
target_dummy = case_data.loc[evaluation_index, "dummy_predicted_reverse"]

# Compare model
print("RF model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_predicted))
print(sklearn.metrics.confusion_matrix(target_actual, target_predicted))
print(sklearn.metrics.accuracy_score(target_actual, target_predicted))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_dummy))
print(sklearn.metrics.confusion_matrix(target_actual, target_dummy))
print(sklearn.metrics.accuracy_score(target_actual, target_dummy))
print("="*32)
print("")

# Output comparison
# Evaluation range
last_century = range(1900, 2000)
evaluation_index = case_data.loc[:, "term"].isin(last_century)
target_actual = case_data.loc[evaluation_index, "case_outcome_reverse"]
target_predicted = case_data.loc[evaluation_index, "rf_predicted_reverse"]
target_dummy = case_data.loc[evaluation_index, "dummy_predicted_reverse"]

# Compare model
print("RF model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_predicted))
print(sklearn.metrics.confusion_matrix(target_actual, target_predicted))
print(sklearn.metrics.accuracy_score(target_actual, target_predicted))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_dummy))
print(sklearn.metrics.confusion_matrix(target_actual, target_dummy))
print(sklearn.metrics.accuracy_score(target_actual, target_dummy))
print("="*32)
print("")

# Output comparison
# Evaluation range
last_century = range(1816, 1900)
evaluation_index = case_data.loc[:, "term"].isin(last_century)
target_actual = case_data.loc[evaluation_index, "case_outcome_reverse"]
target_predicted = case_data.loc[evaluation_index, "rf_predicted_reverse"]
target_dummy = case_data.loc[evaluation_index, "dummy_predicted_reverse"]

# Compare model
print("RF model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_predicted))
print(sklearn.metrics.confusion_matrix(target_actual, target_predicted))
print(sklearn.metrics.accuracy_score(target_actual, target_predicted))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_dummy))
print(sklearn.metrics.confusion_matrix(target_actual, target_dummy))
print(sklearn.metrics.accuracy_score(target_actual, target_dummy))
print("="*32)
print("")

# Setup time series
case_evaluation_index = ~case_data.loc[:, "rf_correct_case"].isnull()
rf_correct_case_ts = case_data.loc[case_evaluation_index, :].groupby("term")["rf_correct_case"].mean()
dummy_correct_case_ts = case_data.loc[case_evaluation_index, :].groupby("term")["dummy_correct_case"].mean()

# Plot all accuracies
f = plt.figure(figsize=(16, 12))
plt.plot(rf_correct_case_ts.index, rf_correct_case_ts,
         marker='o', alpha=0.75)
plt.plot(dummy_correct_case_ts.index, dummy_correct_case_ts,
         marker='>', alpha=0.75)
plt.legend(('Random forest', 'Dummy'))

# Setup time series
rf_spread_case_ts = rf_correct_case_ts - dummy_correct_case_ts

# Plot all accuracies
f = plt.figure(figsize=(16, 12))
plt.bar(rf_spread_case_ts.index, rf_spread_case_ts,
        alpha=0.75)
plt.xlabel("Term")
plt.ylabel("Spread (%)")
plt.title("Spread over dummy model for case accuracy")

# Setup time series
rf_spread_case_dir_ts = pandas.expanding_sum(numpy.sign(rf_spread_case_ts))

# Plot all accuracies
f = plt.figure(figsize=(16, 12))
plt.plot(rf_spread_case_dir_ts.index, rf_spread_case_dir_ts,
        alpha=0.75)

# Output stats
print("t-test:")
print("Uncalibrated:")
print(scipy.stats.ttest_rel(rf_correct_case_ts.values,
                   dummy_correct_case_ts.values))

print("=" * 16)
print("ranksum-test:")
print("Uncalibrated:")
print(scipy.stats.ranksums(rf_correct_case_ts.values,
                   dummy_correct_case_ts.values))

print("=" * 16)
print("Binomial:")
print(statsmodels.stats.proportion.binom_test(case_data["rf_correct_case"].sum(),
                                              case_data["rf_correct_case"].shape[0],
                                              case_data["dummy_correct_case"].mean(),
                                              alternative="larger"))

# Output stats
print("t-test:")
print("Uncalibrated:")
print(scipy.stats.ttest_rel(rf_correct_case_ts.tail(100).values,
                   dummy_correct_case_ts.tail(100).values))

print("=" * 16)
print("ranksum-test:")
print("Uncalibrated:")
print(scipy.stats.ranksums(rf_correct_case_ts.tail(100).values,
                   dummy_correct_case_ts.tail(100).values))

print("=" * 16)
print("Binomial:")
last_century = case_data["term"].drop_duplicates().sort_values().tail(100)
print(statsmodels.stats.proportion.binom_test(case_data.loc[case_data["term"].isin(last_century), "rf_correct_case"].sum(),
                                              case_data.loc[case_data["term"].isin(last_century), "rf_correct_case"].shape[0],
                                              case_data.loc[case_data["term"].isin(last_century), "dummy_correct_case"].mean(),
                                              alternative="larger"))

# Output file data
raw_data.to_csv("../data/output/raw_docket_justice_model_growing_random_forest_5.csv.gz", compression="gzip")
case_data.to_csv("../data/output/raw_docket_case_model_growing_random_forest_5.csv.gz", compression="gzip")
feature_importance_df.to_csv("../data/output/raw_docket_features_model_growing_random_forest_5.csv.gz", compression="gzip")