import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

def create_pipeline(model):
    return model

def evaluate_model_with_feature_selection(model, features, target, cv_strategy):
    custom_f1_scores = cross_val_score(model, features, target, cv=cv_strategy, scoring='f1')
    mean_f1 = np.mean(custom_f1_scores)
    f1_std = np.std(custom_f1_scores)
    return mean_f1, f1_std

def main(file_path):
    data = pd.read_csv(file_path)
    features = data.drop(['MD5', 'label', 'Target'], axis=1)
    target = data['Target']

    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVC": SVC(),
        "BernoulliNB": BernoulliNB()
    }

    # Choose one feature selection method for all models
    selector = SelectKBest(chi2, k=15)

    results = []

    for model_name, model in models.items():
        mean_f1, f1_std = evaluate_model_with_feature_selection(model, features, target, cv_strategy)
        results.append((model_name, mean_f1, f1_std))

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [result[0] for result in results]
    mean_f1_scores = [result[1] for result in results]
    f1_std_devs = [result[2] for result in results]

    x = range(len(models))

    colors = ['b', 'g', 'r', 'c']

    ax.bar(x, mean_f1_scores, yerr=f1_std_devs, align='center', alpha=0.5, ecolor='black', capsize=10, color=colors)
    ax.set_ylabel('F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_title('Model Comparison with Feature Selection')
    ax.yaxis.grid(True)

    plt.show()

    for model_name, mean_f1, f1_std in results:
        print(f"{model_name} - Mean F1: {mean_f1:.4f}, Std: {f1_std:.4f}")

if __name__ == "__main__":
    file_path = 'reduced_dataset.csv'
    main(file_path)
