import nltk
from nltk.tokenize import RegexpTokenizer

def tokenize_text(raw_text):
    tokenizer = RegexpTokenizer(r"(\b\d+\.\d+\b|\w[\w,']*(?:\.\w+)*|\#|\?|!|\.)")
    return tokenizer.tokenize(raw_text)

def sentence_splitting(raw_text):
    lines = raw_text.split('\n', 1)
    if lines[0].isupper(): 
        header = lines[0]
        rest_of_text = lines[1] if len(lines) > 1 else ''
        return [header] + nltk.sent_tokenize(rest_of_text)
    else:
        return nltk.sent_tokenize(raw_text)
