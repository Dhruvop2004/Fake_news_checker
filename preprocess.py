# preprocessing data 


# removing htlm tags
import bs4
from bs4 import BeautifulSoup

def remove_html(text):
    soup=BeautifulSoup(text,'html.parser')
    return soup.get_text(separator='',strip=True)

sources = ['reuters', 'cnn', 'breitbart', 'fox', 'bbc', 'guardian', 
           'politico', 'npr', 'nytimes', 'washingtonpost', 'ap']

def remove_sources(text):
    for s in sources:
        text = re.sub(r'\b' + re.escape(s) + r'\b', '', text)
    return text


# remove punctuations
import re

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)



# lemitization tokenization urls and numbers
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def fast_tokenize(text):
    text = re.sub(r"https?://\S+", "url", text)  # replace URLs
    text = re.sub(r"\d+", "num", text)            # replace numbers
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text) # keep words only
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

def clean_text(text):
    if  not isinstance(text,str):
        return ""
     
    text=text.lower()
    text=remove_html(text)
    text=remove_sources(text)
    text=remove_punctuation(text)
    return text


