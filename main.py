from random_forest_evaluate import *
from preprocessing import *
from embedding import *

def main():
    df = prepareDataset("Dataset_5971.csv")
    df.to_csv("Dataset_5971_cleaned.csv", index=False)

    X_bow, bow_vectorizer = getBowEmbeddings(df['TFIDF_TEXT'], max_features=5000)
    X_tfidf, tfidf_vectorizer = getTfidfEmbeddings(df['TFIDF_TEXT'], max_features=5000)
    X_bow_light, bow_vectorizer_light = getBowEmbeddings(df['TFIDF_TEXT_LIGHT'], max_features=5000)
    X_tfidf_light, tfidf_vectorizer_light = getTfidfEmbeddings(df['TFIDF_TEXT_LIGHT'], max_features=5000)
    feature_names_bow = bow_vectorizer_light.get_feature_names_out()
    X_doc2vec = getDoc2vecEmbeddings(df['TOKENS'].tolist())
    X_word2vec = getWord2vecEmbeddings(df['TOKENS'].tolist())
    X_tfidf_doc2vec = getJoinedEmbeddings(X_tfidf, X_doc2vec)
    X_tfidf_word2vec = getJoinedEmbeddings(X_tfidf, X_word2vec)

    y = df["LABEL_NUM"].values

    results = []

    rf_doc2vec, acc, f1_w, f1_m = evaluate_model(X_doc2vec, y, "Doc2Vec")
    results.append({"Embedding": "Doc2Vec", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_doc2vec = get_confusion_matrix(rf_doc2vec, X_doc2vec, y, "Doc2Vec")

    rf_word2vec, acc, f1_w, f1_m = evaluate_model(X_word2vec, y, "Word2Vec")
    results.append({"Embedding": "Word2Vec", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_word2vec = get_confusion_matrix(rf_word2vec, X_word2vec, y, "Word2Vec")

    rf_tfidf_doc2vec, acc, f1_w, f1_m = evaluate_model(X_tfidf_doc2vec, y, "TFIDF_Doc2Vec")
    results.append({"Embedding": "TFIDF_Doc2Vec", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_tfidf_doc2vec = get_confusion_matrix(rf_tfidf_doc2vec, X_tfidf_doc2vec, y, "TFIDF_Doc2Vec")

    rf_tfidf_word2vec, acc, f1_w, f1_m = evaluate_model(X_tfidf_word2vec, y, "TFIDF_Word2Vec")
    results.append({"Embedding": "TFIDF_Word2Vec", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_tfidf_word2vec = get_confusion_matrix(rf_tfidf_word2vec, X_tfidf_word2vec, y, "TFIDF_Word2Vec")

    rf_tfidf, acc, f1_w, f1_m = evaluate_model(X_tfidf, y, "TFIDF")
    results.append({"Embedding": "TFIDF", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_tfidf = get_confusion_matrix(rf_tfidf, X_tfidf, y, "TFIDF")

    rf_bow, acc, f1_w, f1_m = evaluate_model(X_bow, y, "BoW")
    results.append({"Embedding": "Bow", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_bow = get_confusion_matrix(rf_bow, X_bow, y, "BoW")

    rf_tfidf_light, acc, f1_w, f1_m = evaluate_model(X_tfidf_light, y, "TFIDF_light")
    results.append({"Embedding": "TFIDF_light", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_tfidf_light = get_confusion_matrix(rf_tfidf_light, X_tfidf_light, y, "TFIDF_light")

    rf_bow_light, acc, f1_w, f1_m = evaluate_model(X_bow_light, y, "BoW_light")
    results.append({"Embedding": "Bow_light", "Accuracy": acc, "F1_Weighted": f1_w, "F1_Macro": f1_m})
    cm_df_bow_light = get_confusion_matrix(rf_bow_light, X_bow_light, y, "BoW_light")

    top_features_bow_light = get_top_features(rf_bow_light, feature_names_bow)
    X_bow_dense = X_bow_light.toarray()
    shap_df_bow_light = compute_shap_summary_fast(y, rf_bow_light, X_bow_dense, feature_names_bow)

    results_df = pd.DataFrame(results)
    with pd.ExcelWriter("rf_results_full.xlsx", engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Metrics", index=False)
        top_features_bow_light.to_excel(writer, sheet_name="Top_BoW_Features", index=False)
        cm_df_doc2vec.to_excel(writer, sheet_name="ConfusionMatrix_Doc2Vec", index=True)
        cm_df_word2vec.to_excel(writer, sheet_name="ConfusionMatrix_Word2Vec", index=True)
        cm_df_tfidf_doc2vec.to_excel(writer, sheet_name="ConfusionMatrix_Doc2Vec_TFIDF", index=True)
        cm_df_tfidf_word2vec.to_excel(writer, sheet_name="ConfusionMatrix_Word2Vec_TFIDF", index=True)
        cm_df_tfidf.to_excel(writer, sheet_name="ConfusionMatrix_TFIDF", index=True)
        cm_df_bow.to_excel(writer, sheet_name="ConfusionMatrix_BoW", index=True)
        cm_df_tfidf_light.to_excel(writer, sheet_name="ConfusionMatrix_TFIDF_Light", index=True)
        cm_df_bow_light.to_excel(writer, sheet_name="ConfusionMatrix_BoW_Light", index=True)
        shap_df_bow_light.to_excel(writer, sheet_name="Shap_BoW_Light", index=False)

    print("All results saved to rf_results_full.xlsx")

if __name__ == "__main__":
    main()
