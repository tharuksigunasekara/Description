import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from rake_nltk import Rake


def extract_keywords_tfidf(docs, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    # Get the top keywords for each document
    top_keywords = {}
    for i, doc in enumerate(docs):
        tfidf_scores = tfidf_matrix[i]
        sorted_items = sorted(zip(tfidf_scores.indices, tfidf_scores.data), key=lambda x: x[1], reverse=True)
        top_keywords[i] = [(feature_names[idx], score) for idx, score in sorted_items[:top_n]]

    return top_keywords


def extract_keywords_rake(docs, top_n=10):
    r = Rake()
    keywords = {}

    for i, doc in enumerate(docs):
        r.extract_keywords_from_text(doc)
        rankings = r.get_ranked_phrases_with_scores()
        top_ranked_phrases = rankings[:top_n]
        keywords[i] = top_ranked_phrases

    return keywords


# df = pd.read_csv('../processed_data/processed_dataset_with_entities.csv')
#
# # Extract keywords using TF-IDF for each row in the dataset
# df['Keywords_TFIDF'] = extract_keywords_tfidf(df['Video Description'])
#
# # Extract keywords using RAKE for each row in the dataset
# df['Keywords_RAKE'] = extract_keywords_rake(df['Video Description'])
#
# print(df.head())
#
# # Save the dataframe to a new CSV file
# df.to_csv('processed_data/processed_dataset_with_entities_and_keywords.csv', index=False)
