from scipy import sparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def train_rf(X, y, test_size=0.2, random_state=42, return_model=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_m = f1_score(y_test, y_pred, average="macro")

    if return_model:
        return acc, f1_w, f1_m, rf
    else:
        return acc, f1_w, f1_m


def evaluate_model(X, y, embedding_name, return_model=True):
    acc, f1_w, f1_m, rf = train_rf(X, y, return_model=return_model)
    print(f"{embedding_name} Done: Accuracy={acc:.3f}, F1_macro={f1_m:.3f}")
    return rf, acc, f1_w, f1_m


def get_confusion_matrix(rf, X, y, embedding_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in range(cm.shape[0])],
        columns=[f"Pred_{i}" for i in range(cm.shape[1])]
    )

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {embedding_name}")
    plt.savefig(f"confusion_matrix_{embedding_name}.png", dpi=300)
    plt.show()

    return cm_df


def get_top_features(rf_tfidf, tfidf_feature_names, top_n=20):
    importances = rf_tfidf.feature_importances_
    feat_df = pd.DataFrame({
        "feature": tfidf_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    top20 = feat_df.head(top_n)
    print("\nTop TF-IDF Features:")
    print(top20)
    return top20


def compute_shap_summary_fast(y, model, X_values, feature_names, background_size=50):
    print(f"Computing SHAP values")

    ham_idx = np.random.choice(np.where(y == 0)[0], 450)
    spam_idx = np.random.choice(np.where(y == 1)[0], 450)
    smish_idx = np.random.choice(np.where(y == 2)[0], 450)

    X_balanced_to_explain = X_values[np.concatenate([ham_idx, spam_idx, smish_idx])]

    background_data = shap.utils.sample(X_values, background_size)
    explainer = shap.TreeExplainer(model, data=background_data, feature_perturbation="interventional")

    shap_explanation = explainer(X_balanced_to_explain, check_additivity=False)

    mean_abs_per_class = np.mean(np.abs(shap_explanation.values), axis=0)

    class_names = ["ham", "spam", "smishing"]

    importance_df = pd.DataFrame(
        mean_abs_per_class,
        columns=[f"{c}_SHAP" for c in class_names],
        index=feature_names
    )

    importance_df["Total_Importance"] = importance_df.mean(axis=1)
    importance_df = importance_df.reset_index().rename(columns={'index': 'Feature'})
    importance_df = importance_df.sort_values(by="Total_Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    shap_values_list = [shap_explanation.values[:, :, i] for i in range(len(class_names))]

    shap.summary_plot(
        shap_values_list,
        X_balanced_to_explain,
        plot_type="bar",
        class_names=class_names,
        feature_names=feature_names,
        show=False
    )
    plt.title("Top Predictive Words per Category")
    plt.tight_layout()
    plt.savefig("shap_summary_stacked_bar.png", bbox_inches="tight")
    plt.show()

    return importance_df