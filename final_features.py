import pandas as pd
import requests
import re
import logging
import spacy
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from gensim import corpora
from gensim.models import LdaModel, FastText
from gensim.models.phrases import Phrases, Phraser
from sklearn.decomposition import NMF, PCA
from textblob import TextBlob
import numpy as np
from collections import Counter
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import joblib

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_md", disable=["ner"])
except OSError:
    print("Downloading en_core_web_md...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md", disable=["ner"])

# Custom stopwords
stop_words = set(stopwords.words('english')) - {'no', 'not', 'very', 'few', 'more', 'most', 'against', 'own'}
movement_stopwords = {'nature', 'heart', 'beauty', 'time'}  # Ambiguous across movements
stop_words.update(movement_stopwords)

lemmatizer = WordNetLemmatizer()

# Enhanced movement-specific features with bigrams
movement_specific_features = {
    "Romanticism": [
        "sublime", "melancholy", "solitude", "wilderness", "supernatural", "imagination",
        "pastoral", "medieval", "individualism", "spontaneous", "liberty", "nostalgic",
        "wanderer", "spiritual", "transcendent", "rapture", "reverie", "moonlight",
        "myth", "legend", "yearning", "exotic", "primal", "freedom", "visionary",
        "sublime_nature", "wild_beauty", "lonely_wanderer", "mystic_vision"
    ],
    "Realism": [
        "society", "class", "work", "money", "family", "everyday", "objective", "accurate",
        "ordinary", "urban", "industrial", "scientific", "literal", "concrete", "actual",
        "truthful", "unvarnished", "plain", "prosaic", "commonplace", "labor", "trade",
        "daily_life", "urban_scene", "working_class", "social_order"
    ],
    "Naturalism": [
        "fate", "instinct", "survival", "poverty", "determinism", "environment", "heredity",
        "struggle", "brutal", "primitive", "darwinian", "savage", "squalor", "violence",
        "pessimism", "biological", "amoral", "slum", "tenement", "beast", "natural_selection",
        "grim_reality", "harsh_fate", "social_decay", "animal_instinct"
    ],
    "Gothicism": [
        "shadow", "grave", "fear", "ghost", "horror", "mysterious", "castle", "ruin",
        "terror", "decay", "eerie", "macabre", "haunted", "spectral", "doom", "sinister",
        "crypt", "dungeon", "villain", "specter", "vampire", "coffin", "graveyard",
        "dark_castle", "ghostly_presence", "ancient_curse", "eerie_silence"
    ],
    "Transcendentalism": [
        "spirit", "divine", "self_reliance", "intuition", "oversoul", "harmony", "moral",
        "simplicity", "spiritual", "transcend", "enlightenment", "wilderness", "solitude",
        "contemplation", "insight", "wisdom", "awakening", "self_knowledge", "cosmic_unity",
        "inner_peace", "natural_harmony", "divine_spirit"
    ],
    "Modernism": [
        "fragment", "chaos", "consciousness", "alienation", "subjective", "psychological",
        "disillusion", "existential", "abstract", "innovation", "urban", "relativity",
        "skepticism", "irony", "metropolis", "psychoanalysis", "symbol", "world_war",
        "stream_consciousness", "urban_alienation", "broken_narrative", "modern_anxiety"
    ],
    "Renaissance": [
        "humanism", "virtue", "rhetoric", "reason", "classical", "courtly", "sonnet",
        "wit", "allegory", "pastoral", "knowledge", "patronage", "rebirth", "proportion",
        "florence", "medici", "scholar", "vernacular", "civic_duty", "classical_ideal",
        "courtly_love", "learned_discourse"
    ]
}

# Temporal markers
temporal_markers = {
    "Renaissance": [
        "thee", "thou", "thy", "hath", "doth", "art", "wert", "hast", "forsooth", "methinks"
    ],
    "Romanticism/Gothic": [
        "shall", "upon", "ere", "whilst", "'tis", "o'er", "amidst", "anon", "oft", "alas"
    ],
    "Victorian": [
        "should", "would", "ought", "must", "quite", "indeed", "perhaps", "rather"
    ],
    "Modern": [
        "isn't", "don't", "can't", "won't", "yeah", "okay", "like", "whatever"
    ]
}

def calculate_readability_metrics(text):
    """
    Calculate readability metrics including Dale-Chall.
    """
    try:
        sentences = sent_tokenize(text[:100000])
        if not sentences:
            return {
                "flesch_reading_ease": 0, "gunning_fog": 0, "smog_index": 0,
                "automated_readability_index": 0, "coleman_liau_index": 0, "dale_chall": 0
            }
        
        words = [word.lower() for sentence in sentences for word in sentence.split() if word.strip()]
        if not words:
            return {
                "flesch_reading_ease": 0, "gunning_fog": 0, "smog_index": 0,
                "automated_readability_index": 0, "coleman_liau_index": 0, "dale_chall": 0
            }
        
        word_count = len(words)
        syllable_counts = [count_syllables(word) for word in words]
        complex_words = sum(1 for count in syllable_counts if count > 2)
        
        total_chars = sum(len(word) for word in words)
        avg_sentence_length = word_count / len(sentences)
        avg_syllables_per_word = sum(syllable_counts) / word_count
        
        # Dale-Chall: Use a simple approximation
        difficult_words = sum(1 for word in set(words) if count_syllables(word) > 2 and word not in stop_words)
        dale_chall = 0.1579 * (100 * difficult_words / word_count) + 0.0496 * avg_sentence_length
        if difficult_words / word_count > 0.05:
            dale_chall += 3.6365
        
        flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        gunning_fog = 0.4 * (avg_sentence_length + (100 * complex_words / word_count))
        smog = 1.043 * np.sqrt(30 * complex_words / len(sentences)) + 3.1291
        ari = 4.71 * (total_chars / word_count) + 0.5 * (word_count / len(sentences)) - 21.43
        L = (total_chars / word_count) * 100
        S = (len(sentences) / word_count) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        
        return {
            "flesch_reading_ease": flesch, "gunning_fog": gunning_fog, "smog_index": smog,
            "automated_readability_index": ari, "coleman_liau_index": coleman_liau, "dale_chall": dale_chall
        }
    except Exception as e:
        logger.error(f"Error in readability calculation: {e}")
        return {
            "flesch_reading_ease": 0, "gunning_fog": 0, "smog_index": 0,
            "automated_readability_index": 0, "coleman_liau_index": 0, "dale_chall": 0
        }

def extract_syntactic_features(text):
    """
    Extract syntactic features with dependency parsing.
    """
    sentences = sent_tokenize(text[:50000])
    if not sentences:
        return {}
    
    sentence_lengths = [len(sent.split()) for sent in sentences]
    
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    var_sentence_length = np.var(sentence_lengths) if sentence_lengths else 0
    max_sentence_length = max(sentence_lengths) if sentence_lengths else 0
    median_sentence_length = np.median(sentence_lengths) if sentence_lengths else 0
    
    doc = nlp(text[:50000])
    sentence_count = len(sentences) or 1
    
    punctuation_counts = {
        "exclamation_marks": text.count("!") / sentence_count,
        "question_marks": text.count("?") / sentence_count,
        "semicolons": text.count(";") / sentence_count,
        "colons": text.count(":") / sentence_count,
        "ellipses": (text.count("...") + text.count(". . .")) / sentence_count,
        "dashes": (text.count("—") + text.count("--") + text.count("–")) / sentence_count
    }
    
    pos_counts = Counter(token.pos_ for token in doc)
    total_tokens = sum(pos_counts.values()) or 1
    
    pos_features = {
        "noun_freq": pos_counts.get("NOUN", 0) / total_tokens * 100,
        "adj_freq": pos_counts.get("ADJ", 0) / total_tokens * 100,
        "verb_freq": pos_counts.get("VERB", 0) / total_tokens * 100,
        "prep_freq": pos_counts.get("ADP", 0) / total_tokens * 100
    }
    
    dep_features = {
        "appositive_freq": sum(1 for token in doc if token.dep_ == "appos") / total_tokens * 100,
        "relative_clause_freq": sum(1 for token in doc if token.dep_ in ["relcl"]) / total_tokens * 100
    }
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "var_sentence_length": var_sentence_length,
        "max_sentence_length": max_sentence_length,
        "median_sentence_length": median_sentence_length,
        **punctuation_counts,
        **pos_features,
        **dep_features
    }
# Function to count syllables in a word with improved accuracy
def count_syllables(word):
    try:
        word = word.lower().strip()
        if not word:
            return 0
        
        special_cases = {
            'the': 1, 'every': 2, 'different': 3, 'difficult': 3, 'beautiful': 3,
            'absolutely': 4, 'area': 2, 'business': 2, 'camera': 3, 'poem': 2,
            'poems': 1, 'through': 1, 'thoroughly': 3, 'thought': 1, 'variable': 4,
            'society': 3, 'science': 2
        }
        
        if word in special_cases:
            return special_cases[word]
        
        if word.endswith('es') and not word.endswith(('ies', 'ves', 'oes')):
            word = word[:-2]
        elif word.endswith('ed') and not word.endswith('ied'):
            word = word[:-2]
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False
        
        for i, char in enumerate(word):
            is_vowel = char in vowels
            
            if is_vowel and prev_char_was_vowel and i > 0:
                if word[i-1:i+1] not in ['io', 'ia', 'eo', 'ua', 'uo', 'ie']:
                    syllable_count += 1
            elif is_vowel and not prev_char_was_vowel:
                syllable_count += 1
                
            prev_char_was_vowel = is_vowel
        
        if word.endswith('e') and not word.endswith(('le', 'ie', 'ee', 'oe')):
            syllable_count = max(1, syllable_count - 1)
        
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
            
        if word.endswith('y') and len(word) > 1 and word[-2] not in vowels:
            syllable_count += 1
        
        return max(1, syllable_count)
    except Exception as e:
        logger.error(f"Error counting syllables for word '{word}': {e}")
        return 1

def preprocess_text(text, for_vectorizer=True):
    """
    Preprocess text, preserving stylistic elements.
    """
    try:
        doc = nlp(text[:50000])
        sentences = sent_tokenize(text[:50000])
        tokens_per_sentence = []
        for sentence in sentences:
            sent_doc = nlp(sentence)
            tokens = [
                token.text.lower() if for_vectorizer else token.text
                for token in sent_doc
                if token.is_alpha and not token.is_stop and token.text.lower() not in stop_words
            ]
            if len(tokens) > 2:
                tokens_per_sentence.append(tokens)
        
        if for_vectorizer:
            return " ".join([token for sent in tokens_per_sentence for token in sent])
        return tokens_per_sentence
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return "" if for_vectorizer else []
# Function to download and clean Gutenberg text
def download_gutenberg_text(gid):
    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(5):
        for url in urls:
            try:
                response = requests.get(url, timeout=20, headers=headers)
                response.raise_for_status()
                text = response.text
                
                start_markers = [
                    "*** START OF THIS PROJECT GUTENBERG EBOOK",
                    "***START OF THE PROJECT GUTENBERG EBOOK",
                    "*** START OF THE PROJECT GUTENBERG EBOOK",
                    "***START OF THIS PROJECT GUTENBERG EBOOK",
                    "START OF THE PROJECT GUTENBERG EBOOK",
                    "START OF THIS PROJECT GUTENBERG EBOOK",
                    "*** START OF THE PROJECT GUTENBERG",
                    "*** START OF THIS PROJECT GUTENBERG",
                    "Produced by",
                    "Transcribed from",
                    "This eBook was produced by"
                ]
                
                end_markers = [
                    "*** END OF THIS PROJECT GUTENBERG EBOOK",
                    "***END OF THE PROJECT GUTENBERG EBOOK",
                    "*** END OF THE PROJECT GUTENBERG EBOOK",
                    "***END OF THIS PROJECT GUTENBERG EBOOK",
                    "END OF THE PROJECT GUTENBERG EBOOK",
                    "END OF THIS PROJECT GUTENBERG EBOOK",
                    "*** END OF THE PROJECT GUTENBERG",
                    "*** END OF THIS PROJECT GUTENBERG",
                    "End of Project Gutenberg",
                    "End of the Project Gutenberg"
                ]
                
                start_idx = -1
                for marker in start_markers:
                    pos = text.find(marker)
                    if pos != -1:
                        line_end = text.find('\n', pos)
                        if line_end != -1:
                            start_idx = line_end + 1
                        else:
                            start_idx = pos + len(marker)
                        break
                
                end_idx = -1
                for marker in end_markers:
                    pos = text.find(marker)
                    if pos != -1:
                        end_idx = pos
                        break
                
                if start_idx != -1 and end_idx != -1:
                    text = text[start_idx:end_idx].strip()
                elif start_idx != -1:
                    text = text[start_idx:].strip()
                elif end_idx != -1:
                    text = text[:end_idx].strip()
                
                if len(text) > 500000:
                    chapter_markers = [
                        "\nCHAPTER I", "\nCHAPTER 1", "\nI.", "\n1.",
                        "\nPREFACE", "\nINTRODUCTION", "\nFOREWORD"
                    ]
                    for marker in chapter_markers:
                        pos = text.find(marker)
                        if pos > 1000:
                            text = text[pos:]
                            break
                
                text = re.sub(r'\r\n', '\n', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'_+', '', text)
                text = re.sub(r'\[.*?\]', '', text)
                text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
                
                if len(text.split()) > 1000:
                    return text
                
            except requests.RequestException as e:
                logger.warning(f"Failed to download {url}: {e}")
        
        sleep_time = 2 * (2 ** attempt)
        logger.warning(f"Retry {attempt+1}/5, waiting {sleep_time} seconds...")
        time.sleep(sleep_time)
    
    logger.error(f"No valid text found for GutenbergID {gid} after 5 attempts")
    return None

def extract_tfidf_features(texts, max_features=500):
    """
    Extract TF-IDF features with sublinear scaling.
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 3), min_df=2, sublinear_tf=True
        )
        X = vectorizer.fit_transform(texts)
        return X.toarray(), vectorizer.get_feature_names_out(), vectorizer
    except Exception as e:
        logger.error(f"Error in TF-IDF extraction: {e}")
        return np.zeros((len(texts), max_features)), [], None

def extract_lda_features(texts, num_topics=15):
    """
    Extract LDA topic features with TF-IDF input.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
        X = vectorizer.fit_transform([" ".join(sent) for text in texts for sent in text])
        dictionary = corpora.Dictionary([sent for text in texts for sent in text])
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(sent) for text in texts for sent in text]
        lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
        topic_probs = np.zeros((len(texts), num_topics))
        for i, text in enumerate(texts):
            doc = [dictionary.doc2bow(sent) for sent in text]
            topics = [lda[bow] for bow in doc]
            avg_probs = np.mean([[prob for _, prob in topic] for topic in topics], axis=0)
            topic_probs[i, :len(avg_probs)] = avg_probs
        return topic_probs
    except Exception as e:
        logger.error(f"Error in LDA extraction: {e}")
        return np.zeros((len(texts), num_topics))

def extract_nmf_features(texts, num_topics=15):
    """
    Extract NMF topic features.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
        X = vectorizer.fit_transform([" ".join(sent) for text in texts for sent in text])
        nmf = NMF(n_components=num_topics, random_state=42)
        W = nmf.fit_transform(X)
        return W[:len(texts)]
    except Exception as e:
        logger.error(f"Error in NMF extraction: {e}")
        return np.zeros((len(texts), num_topics))

def extract_embedding_features(texts):
    """
    Extract mean GloVe embeddings from SpaCy.
    """
    try:
        embeddings = np.zeros((len(texts), 300))  # en_core_web_md uses 300D vectors
        for i, text in enumerate(texts):
            doc = nlp(" ".join([token for sent in text for token in sent])[:10000])
            valid_tokens = [token for token in doc if token.has_vector]
            if valid_tokens:
                embeddings[i] = np.mean([token.vector for token in valid_tokens], axis=0)
        return embeddings
    except Exception as e:
        logger.error(f"Error in embedding extraction: {e}")
        return np.zeros((len(texts), 300))

def extract_movement_term_features(text):
    """
    Extract TF-IDF-weighted movement-specific term frequencies.
    """
    features = {}
    doc = nlp(text[:50000])
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    total_tokens = len(tokens) or 1
    
    # Create bigrams
    bigram_model = Phrases([tokens], min_count=2, threshold=10)
    bigram_phraser = Phraser(bigram_model)
    bigrams = bigram_phraser[tokens]
    
    # TF-IDF weighting
    vectorizer = TfidfVectorizer(vocabulary=[term for terms in movement_specific_features.values() for term in terms])
    tfidf_matrix = vectorizer.fit_transform([" ".join(bigrams)])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    
    for movement, terms in movement_specific_features.items():
        term_score = sum(tfidf_scores.get(term, 0) for term in terms)
        features[f"{movement}_term_score"] = term_score
    
    return features

def extract_temporal_features(text):
    """
    Extract temporal marker frequencies.
    """
    features = {}
    doc = nlp(text[:50000])
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    total_tokens = len(tokens) or 1
    
    for era, markers in temporal_markers.items():
        marker_count = sum(tokens.count(marker) for marker in markers)
        features[f"{era}_marker_freq"] = marker_count / total_tokens * 100
    
    return features

def extract_sentiment_features(text):
    """
    Extract sentiment polarity and subjectivity.
    """
    try:
        blob = TextBlob(text[:50000])
        return {
            "sentiment_polarity": blob.sentiment.polarity,
            "sentiment_subjectivity": blob.sentiment.subjectivity
        }
    except Exception as e:
        logger.error(f"Error in sentiment extraction: {e}")
        return {"sentiment_polarity": 0, "sentiment_subjectivity": 0}

def select_features(X, y, feature_names, k=200):
    """
    Select top k features using mutual_info_classif.
    """
    try:
        selector = SelectKBest(mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_names = feature_names[selector.get_support()]
        
        # Apply PCA
        pca = PCA(n_components=min(50, X_selected.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_selected)
        pca_names = [f"pca_{i}" for i in range(X_pca.shape[1])]
        
        # Log feature statistics
        logger.info(f"Selected {len(selected_names)} features: {', '.join(selected_names[:10])}...")
        logger.info(f"PCA reduced to {X_pca.shape[1]} components, explained variance: {sum(pca.explained_variance_ratio_):.2%}")
        
        return X_pca, np.array(pca_names)
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        return X, feature_names

def process_texts(metadata_file, output_file="edited_text_features.csv", max_texts=None):
    """
    Process texts and extract features, saving to CSV.
    """
    try:
        # Load metadata
        df = pd.read_csv(metadata_file)
        if max_texts:
            df = df[:max_texts]
        
        logger.info(f"Processing {len(df)} texts")
        
        # Cache directory
        cache_dir = "data/text_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        texts = []
        for _, row in df.iterrows():
            gid = row['GutenbergID']
            cache_file = os.path.join(cache_dir, f"{gid}.txt")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                logger.info(f"Loaded cached text for GutenbergID {gid}")
            else:
                text = download_gutenberg_text(gid)
                if text:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    logger.info(f"Cached text for GutenbergID {gid}")
                else:
                    text = ""
            texts.append(text)
        
        # Preprocess texts
        preprocessed_texts = [preprocess_text(text, for_vectorizer=True) for text in texts]
        tokenized_texts = [preprocess_text(text, for_vectorizer=False) for text in texts]
        
        # Extract features
        features = []
        feature_names = []
        
        # TF-IDF features
        tfidf_features, tfidf_names, tfidf_vectorizer = extract_tfidf_features(preprocessed_texts)
        features.append(tfidf_features)
        feature_names.extend([f"tfidf_{name}" for name in tfidf_names])
        
        # LDA features
        lda_features = extract_lda_features(tokenized_texts)
        features.append(lda_features)
        feature_names.extend([f"lda_topic_{i}" for i in range(lda_features.shape[1])])
        
        # NMF features
        nmf_features = extract_nmf_features(tokenized_texts)
        features.append(nmf_features)
        feature_names.extend([f"nmf_topic_{i}" for i in range(nmf_features.shape[1])])
        
        # GloVe embeddings
        embedding_features = extract_embedding_features(tokenized_texts)
        features.append(embedding_features)
        feature_names.extend([f"glove_{i}" for i in range(embedding_features.shape[1])])
        
        # Per-text features
        for i, text in enumerate(texts):
            if not text:
                continue
            text_features = {}
            
            # Readability metrics
            text_features.update(calculate_readability_metrics(text))
            
            # Syntactic features
            text_features.update(extract_syntactic_features(text))
            
            # Movement-specific terms
            text_features.update(extract_movement_term_features(text))
            
            # Temporal markers
            text_features.update(extract_temporal_features(text))
            
            # Sentiment
            text_features.update(extract_sentiment_features(text))
            
            features.append(np.array([list(text_features.values())]))
            if i == 0:
                feature_names.extend(list(text_features.keys()))
        
        # Combine features
        X = np.hstack([f for f in features if f.shape[0] == len(texts)])
        
        # Feature selection
        y = df['Movement'].values
        X_selected, selected_names = select_features(X, y, np.array(feature_names))
        
        # Create DataFrame
        feature_df = pd.DataFrame(X_selected, columns=selected_names)
        feature_df['GutenbergID'] = df['GutenbergID']
        feature_df['Author'] = df['Author']
        feature_df['Movement'] = df['Movement']
        
        # Save to CSV
        feature_df.to_csv(output_file, index=False)
        logger.info(f"Saved features to {output_file}")
        
        # Save vectorizer and PCA
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        joblib.dump(PCA, 'pca_model.joblib')
        
    except Exception as e:
        logger.error(f"Error in processing texts: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features from Gutenberg texts")
    parser.add_argument('--metadata', type=str, default='data/data.csv', help='Path to metadata CSV')
    parser.add_argument('--output', type=str, default='text_features.csv', help='Output CSV file')
    parser.add_argument('--max_texts', type=int, default=None, help='Maximum number of texts to process')
    args = parser.parse_args()
    
    process_texts(args.metadata, args.output, args.max_texts)