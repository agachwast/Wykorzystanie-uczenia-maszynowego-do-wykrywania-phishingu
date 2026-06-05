import pandas as pd
from sklearn.model_selection import train_test_split

from random_forest_evaluate import *
from preprocessing import *
from embedding import *

def run_experiment(X, y, name, results):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf, acc, f1_w, f1_m = evaluate_model(
        X_train, X_test, y_train, y_test, name
    )

    results.append({
        "Embedding": name,
        "Accuracy": acc,
        "F1_Weighted": f1_w,
        "F1_Macro": f1_m
    })

    cm_df = get_confusion_matrix(rf, X_test, y_test, name)

    return rf, cm_df

def main():

    df = prepareDataset("Dataset_5971.csv")
    df.to_csv("Dataset_5971_cleaned.csv", index=False)

    y = df["LABEL_NUM"].values

    X_bow, bow_vectorizer = getBowEmbeddings(df['TFIDF_TEXT'], max_features=5000)
    X_tfidf, tfidf_vectorizer = getTfidfEmbeddings(df['TFIDF_TEXT'], max_features=5000)
    X_bow_light, bow_vectorizer_light = getBowEmbeddings(df['TFIDF_TEXT_LIGHT'], max_features=5000)
    X_tfidf_light, tfidf_vectorizer_light = getTfidfEmbeddings(df['TFIDF_TEXT_LIGHT'], max_features=5000)

    X_doc2vec = getDoc2vecEmbeddings(df['TOKENS'].tolist())
    X_word2vec = getWord2vecEmbeddings(df['TOKENS'].tolist())

    X_tfidf_doc2vec = getJoinedEmbeddings(X_tfidf, X_doc2vec)
    X_tfidf_word2vec = getJoinedEmbeddings(X_tfidf, X_word2vec)

    feature_names = bow_vectorizer_light.get_feature_names_out()

    results = []

    rf_doc2vec, cm_doc2vec = run_experiment(X_doc2vec, y, "Doc2Vec", results)
    rf_word2vec, cm_word2vec = run_experiment(X_word2vec, y, "Word2Vec", results)

    rf_tfidf_doc2vec, cm_td = run_experiment(X_tfidf_doc2vec, y, "TFIDF_Doc2Vec", results)
    rf_tfidf_word2vec, cm_tw = run_experiment(X_tfidf_word2vec, y, "TFIDF_Word2Vec", results)

    rf_tfidf, cm_tfidf = run_experiment(X_tfidf, y, "TFIDF", results)
    rf_bow, cm_bow = run_experiment(X_bow, y, "BoW", results)

    rf_tfidf_light, cm_tfidf_light = run_experiment(X_tfidf_light, y, "TFIDF_light", results)
    rf_bow_light, cm_bow_light = run_experiment(X_bow_light, y, "BoW_light", results)

    # --- SHAP (tylko BoW light) ---
    top_features_bow_light = get_top_features(rf_bow_light, feature_names)

    X_bow_dense = X_bow_light.toarray()
    shap_df_bow_light = compute_shap_summary_fast(
        y, rf_bow_light, X_bow_dense, feature_names
    )

    results_df = pd.DataFrame(results)

    with pd.ExcelWriter("rf_results_full.xlsx", engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Metrics", index=False)

        top_features_bow_light.to_excel(writer, sheet_name="Top_BoW_Features", index=False)

        cm_doc2vec.to_excel(writer, sheet_name="CM_Doc2Vec")
        cm_word2vec.to_excel(writer, sheet_name="CM_Word2Vec")
        cm_td.to_excel(writer, sheet_name="CM_TFIDF_Doc2Vec")
        cm_tw.to_excel(writer, sheet_name="CM_TFIDF_Word2Vec")
        cm_tfidf.to_excel(writer, sheet_name="CM_TFIDF")
        cm_bow.to_excel(writer, sheet_name="CM_BoW")
        cm_tfidf_light.to_excel(writer, sheet_name="CM_TFIDF_Light")
        cm_bow_light.to_excel(writer, sheet_name="CM_BoW_Light")

        shap_df_bow_light.to_excel(writer, sheet_name="SHAP_BoW_Light", index=False)

    print("All results saved to rf_results_full.xlsx")


if __name__ == "__main__":
    main()
