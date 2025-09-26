import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
def tokenize_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens
# Sanitizes text from punctuation and stopwords
def clean_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    cleaned_tokens =[]
    for token in tokens:
        if token not in string.punctuation and token not in stop_words:
            cleaned_tokens.append(token)
    return cleaned_tokens
def normalize_tokens(tokens,option):
    if option == 'l':
        lemmatizer = WordNetLemmatizer()
        normalized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    elif option== 's':
        stemmer = PorterStemmer()
        normalized_tokens = [stemmer.stem(token) for token in tokens]
    else:
        raise ValueError("Invalid option. Use 'l' for lemmatization or 's' for stemming.")
    return normalized_tokens
def preprocess_text(text):
    tokens = tokenize_text(text)
    cleaned_tokens = clean_tokens(tokens)
    normalized_tokens = normalize_tokens(cleaned_tokens, option='l')
    cleaned_text = ' '.join(normalized_tokens)
    return cleaned_text
def preprocess_paper(paper):
    text_paper_metadata = f"{paper['Title']} {paper['Authors']} {paper['Abstract']} {paper['Subject_Tags']} {paper['Subjects']} {paper['Submitted Date']}"
    return preprocess_text(text_paper_metadata)

if __name__ == "__main__":
    sample_text = "Artificial intelligence and machine learning techniques are advancing rapidly in 2023!"

    print("=== Original Text ===")
    print(sample_text)

    # B1: tokenize
    tokens = tokenize_text(sample_text)
    print("\n--- Tokens ---")
    print(tokens)

    # B2: clean
    cleaned = clean_tokens(tokens)
    print("\n--- Cleaned Tokens (no stopwords/punct) ---")
    print(cleaned)

    # B3: normalize (lemmatize hoặc stem)
    normalized_l = normalize_tokens(cleaned, option='l')
    print("\n--- Lemmatized Tokens ---")
    print(normalized_l)

    normalized_s = normalize_tokens(cleaned, option='s')
    print("\n--- Stemmed Tokens ---")
    print(normalized_s)

    # B4: full preprocess_text
    preprocessed_text = preprocess_text(sample_text)
    print("\n=== Final Preprocessed Text ===")
    print(preprocessed_text)

    # Demo cho paper
    paper = {
        "Title": "Deep Learning for NLP",
        "Authors": "Alice, Bob",
        "Abstract": "This paper introduces a new approach for text classification.",
        "Subject_Tags": "AI, NLP",
        "Subjects": "Computer Science",
        "Submitted Date": "2023-08-01"
    }

    print("\n=== Preprocess Paper Metadata ===")
    print(preprocess_paper(paper))
