from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def tokenize(text):
    """Tokenize a a given message. We assume the messages contain only sentence and dont split the message into multiple sentences

    Args:
    text: string. Input message to tokenize

    Returns:
    tokens: List of tokens in the given text after removing stop words, and lemmatization
    """    
    
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