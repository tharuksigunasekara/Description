import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from html import unescape


# Ensure required NLTK resources are downloaded
def download_nltk_resources():
    resources = [
        ('wordnet', 'corpora/wordnet'),
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords')
    ]

    for resource, path in resources:
        try:
            nltk.data.find(path)
            print(f'{resource} is already downloaded.')
        except LookupError:
            nltk.download(resource)


# download_nltk_resources()


def preprocess_text(text):
    text = unescape(text)  # HTML unescape
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove email addresses
    text = re.sub(r'@\S+', '', text)  # Remove handles
    text, hashtags = extract_hashtags(text)  # Extract hashtags
    text = remove_emojis(text)  # Remove emojis
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
    text = re.sub(r"[^a-zA-Z#]", " ", text)  # Remove punctuations, numbers, special chars
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if
              word not in set(stopwords.words('english')) and len(word) > 3]  # Remove stopwords and short words
    lemmatizer = WordNetLemmatizer()  # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens), hashtags  # Re-join tokens


def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    text = re.sub(r'#\w+', '', text)  # Remove hashtags from text
    return text, hashtags


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)  # Remove emoji

# # Load data
# df = pd.read_csv('dataset/dataset.csv', encoding='ISO-8859-1')
# df = df.dropna(subset=['Video Description'])  # remove null values
#
# # Apply preprocessing to each description
# df['processed_Video Description'] = df['Video Description'].apply(preprocess_text)
#
# # Save the processed data
# df.to_csv('processed_data/processed_dataset.csv', index=False)
