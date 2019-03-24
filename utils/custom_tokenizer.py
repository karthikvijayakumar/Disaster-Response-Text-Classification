from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def tokenize(text):
    # Note that the tweets are largely one sentence tweets, hence we dont need to split into multiple sentences
    
    #Remove non alpha numeric characters
    text = re.sub('[^a-zA-Z0-9]', ' ', text.lower())
    
    # Compress multiple spaces into one
    text = re.sub( '[ ]+', ' ',  text )
    
    #Use word tokenizer from NLTK
    tokens = word_tokenize(text)
    
    #Remove stop words
    english_stop_words = stopwords.words("english")
    tokens = filter(lambda x: x not in english_stop_words, tokens)
    
    #Lemmatize using the Wordnet Lemmatizer
    lemmatizer =  WordNetLemmatizer()
    
    tokens = [x.strip() for x in tokens]
    tokens = [lemmatizer.lemmatize(x) for x in tokens]
    tokens = [lemmatizer.lemmatize(x, pos= 'v') for x in tokens]
    
    return tokens

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens