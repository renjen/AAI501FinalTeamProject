# Project Plan: Low-Cost Human Activity Recognition

## Problem statement

An elder-care or fitness app needs to know what a user is doing (walking,
walking upstairs or downstairs, sitting, standing, lying down) from a phone
that runs all day on battery. Standard classifiers already reach about 96%
accuracy on this dataset, so accuracy is not our headline question.
We study **how cheaply** activity can be recognized and **how well it works
for new users**:

1. **Which models work best, and why?** (classification, tuning)
2. **How few features and sensors do we need?** (feature selection;
   accelerometer vs. gyroscope, since the gyroscope uses more battery)
3. **What structure exists without labels, and who does the model fail
   for?** (clustering, per-person generalization)

**Dataset:** UCI HAR Using Smartphones. 10,299 windows, 561 features,
30 people, 6 classes. Official split: 21 people for training, 9 for testing.

**Course topics covered:** classification, clustering, feature selection,
hyperparameter tuning, dimensionality reduction (PCA), and deep learning
(optional).

## Ground rules (everyone)

- Load data **only** through `src/har_data.py` so everyone uses the same
  split.
- **Never tune on the test set.** Use cross-validation on the training set
  for tuning. Use `GroupKFold` with `groups=subject_train` so that one
  person's data never appears in both training and validation folds.
- Use `random_state=42` everywhere.
- Report **macro F1** plus accuracy, and save a confusion matrix for every
  final model.
- Save results as CSV in `results/` and figures as PNG in `figures/`, named
  `<person>_<topic>.<ext>` (for example `B_accuracy_vs_features.png`).
- Code follows PEP 8. Give every function a docstring.
- Each person works on their own branch (`person-a`, `person-b`, `person-c`)
  and opens a pull request to `main`. Someone else reviews it. Commit often,
  because commit history is how collaboration is measured.

## Person A: Classifier comparison and error analysis

**Notebook:** `notebooks/A_classifiers.ipynb`

1. **Exploratory data analysis (shared section of the report).** Class
   balance, samples per person, and a 2-D PCA or t-SNE plot colored by
   activity.
2. **Baselines.** Logistic regression, k-nearest neighbors, decision tree,
   random forest, linear SVM, RBF SVM, and gradient boosting, each with
   default settings.
3. **Tuning.** `GridSearchCV` with `GroupKFold` on the top 3 to 4 models
   (for example SVM `C` and `gamma`, random forest `n_estimators` and
   `max_depth`, kNN `k`). Plot validation score against each key
   parameter.
4. **Final comparison.** One table and one bar chart (accuracy, macro F1,
   training time, prediction time) on the test set.
5. **Error analysis.** Confusion matrices. Look in depth at **sitting vs.
   standing**: which features separate the two, and why the phone struggles
   with this pair.

**Deliverables:** model comparison table and chart, tuning curves,
confusion matrices, and the sitting vs. standing analysis.

## Person B: Feature selection and sensor cost

**Notebook:** `notebooks/B_feature_selection.ipynb`

1. **Rank features three ways.** Random forest importance, mutual
   information (`SelectKBest`), and L1-regularized logistic regression.
   Compare how much the three top-20 lists overlap.
2. **Accuracy vs. number of features.** For k = 5, 10, 20, 50, 100, 200,
   and 561, retrain Person A's best model and plot accuracy against k for
   each ranking method. Identify where accuracy levels off.
3. **Recursive feature elimination** (`RFECV`) as a fourth method, if time
   allows.
4. **Sensor comparison.** Accelerometer only (348 features), gyroscope only
   (213), and both (561), using `sensor_masks()`. Early test with a linear
   SVM: 91.4%, 82.5%, and 96.7%.
5. **PCA vs. feature selection.** Accuracy with the top k principal
   components vs. the top k selected features (selected features are easier
   to interpret).
6. **Takeaway.** For example: "Accelerometer only with N features keeps X%
   accuracy," which supports a cheaper app.

**Deliverables:** accuracy-vs-features plot, sensor comparison chart,
top-feature table with plain-English meanings, and the recommendation.

## Person C: Clustering and generalization to new people

**Notebook:** `notebooks/C_clustering_generalization.ipynb`

1. **Clustering without labels.** Run k-means, a Gaussian mixture model,
   and agglomerative clustering on standardized features (after PCA).
   Choose k with the elbow method and silhouette score. Compare clusters to
   the true labels using the adjusted Rand index (ARI). Early result:
   k = 2 separates moving from still almost perfectly (ARI 0.99), while
   k = 6 reaches only 0.46.
2. **Interpret the clusters.** Which activities merge together? Show a
   cluster vs. activity table and a PCA scatter plot.
3. **Generalization to new people.** Accuracy for each of the 9 test
   people (early test: 88.8% to 99.7%). Run leave-one-person-out
   cross-validation on the training set.
4. **Leakage experiment.** Compare a random 70/30 split, which mixes the
   same people into training and test, with the person-based split. Show
   that the random split inflates the scores.
5. **(Optional) Deep learning.** A 1D CNN or LSTM on the raw signals from
   `load_raw_signals()`, shape (n, 128, 9), compared with the engineered
   features. This tests whether hand-built features are needed.

**Deliverables:** clustering comparison and plots, per-person accuracy
chart, leakage comparison, and the optional CNN result.

## Timeline

| Week | Everyone | A | B | C |
|---|---|---|---|---|
| 1–2 | Set up the repo and environment. Read the dataset paper. | Exploratory data analysis | Read about feature selection | Read about clustering for HAR |
| 3 | **Submit the proposal (1–2 pages).** Each person writes about their own section. | Baselines | Feature rankings | First clustering results |
| 4 | Weekly check-in, review PRs | Tuning | Accuracy vs. features, sensor comparison | Clustering comparison, per-person accuracy |
| 5 | Freeze results | Final comparison, error analysis | RFE, PCA comparison | Leakage experiment, optional CNN |
| 6 | Write the report (about 10 pages, APA 7). Each person writes about their own section. | Methods and results for classifiers | Feature selection | Clustering and generalization |
| 7 | Record the presentation (20–30 minutes, about 8 minutes each). Submit. | Intro and classifiers | Feature selection | Clustering and conclusions |

Shared writing: introduction and business problem, related work, and
conclusion and next steps. Assign these at the week 5 check-in.

## References (starting list, APA 7)

- Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).
  A public domain dataset for human activity recognition using smartphones.
  *21st European Symposium on Artificial Neural Networks (ESANN)*.
- Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
  *Human activity recognition using smartphones* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C54S4K
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research, 12*, 2825–2830.
