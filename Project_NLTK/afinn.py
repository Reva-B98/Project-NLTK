import math
import re


filenameAFINN = '/Users/revabalasundaram/Documents/COMP 6591 - NLA/VS/Project_NLTK/AFINN-111.txt'
with open(filenameAFINN, encoding='utf-8') as file:
    afinn = dict(map(lambda ws: (ws[0], int(ws[1])), [ws.strip().split('\t') for ws in file]))


pattern_split = re.compile(r"\W+")

def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence. 
    """
    words = pattern_split.split(text.lower())
    sentiments = list(map(lambda word: afinn.get(word, 0), words))
    if sentiments:
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
    else:
        sentiment = 0
    return sentiment

if __name__ == '__main__':

    text = "The film, lacking real acting and originality, is not a failure"
    print("%6.2f %s" % (sentiment(text), text))
    

    text = "john carpenter apparently believes that action scenes in which people fight something horrible are the same as horror scenes. For a writer and director of horror films , supposedly an expert on horror , it is a very bad mistake to make"
    print("%6.2f %s" % (sentiment(text), text))
