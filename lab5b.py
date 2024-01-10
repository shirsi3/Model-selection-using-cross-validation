import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_classif
from sklearn.metrics import make_scorer, f1_score

#feature selector and an SVM classifier
def create_pipeline(selector):
    return Pipeline([
        ('feature_selector', selector),
        ('svc_classifier', SVC())
    ])

# crossvalidation feature selection methods
def crossvalidation_feature(file_path, cv_splits=10):

    dataframe = pd.read_csv(file_path)
    
    features = dataframe.drop(columns=['MD5', 'label', 'Target'])
    target = dataframe['Target']
    # Create a cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    #dfeature selection methods
    feature_selection = {
        "SelectKBest Chi2": SelectKBest(chi2, k=15),
        "SelectKBest Mutual info": SelectKBest(mutual_info_classif, k=15),
        "SelectPercentile Chi2": SelectPercentile(chi2, percentile=10),
        "SelectPercentile Mutual info": SelectPercentile(mutual_info_classif, percentile=10)
    }

    for feature_method, selector in feature_selection.items():
        #  pipeline with the feature selector and classifier
        pipeline = create_pipeline(selector)
       
        classification_scorer = make_scorer(f1_score)
        #cross-validation and get F1 scores
        evaluation_scores = cross_val_score(pipeline, features, target, cv=cv_strategy, scoring=classification_scorer)
        #  mean and standard deviation of F1 scores
        mean_f1 = np.mean(evaluation_scores)
        f1_std = np.std(evaluation_scores)
        # Print 
        print(f"{feature_method} - Mean F1: {mean_f1}, Std: {f1_std}")

if __name__ == "__main__":
    file_path = 'reduced_dataset.csv'
    crossvalidation_feature(file_path)
