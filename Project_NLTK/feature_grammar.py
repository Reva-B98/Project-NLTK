import nltk
from nltk.grammar import FeatureGrammar
from nltk.parse import FeatureEarleyChartParser
from nltk.corpus import opinion_lexicon
from nltk.corpus import movie_reviews
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

nltk.download('maxent_ne_chunker')
nltk.download('words')

import random
def featuregrammar(sentences,entity):

    def extract_overall_sentiment(tree):
        if 'SENTIMENT' in tree.label():
            return tree.label()['SENTIMENT']
        
    def extract_entity_sentiment(tree, entity):
        for subtree in tree.subtrees():
            if entity in subtree.leaves():
                current = subtree
                while current and current.label() != 'S':
                    if 'SENTIMENT' in current.label():
                        return current.label()['SENTIMENT']
                    current = current.parent()
        return None
    def preprocess_sentence(sentence):
        processed_sentence = []
        for word in sentence:
            if word in opinion_lexicon.positive():
                sentiment = 'positive'
                processed_sentence.append(f"{word}[SENTIMENT={sentiment}]")
            elif word in opinion_lexicon.negative():
                sentiment = 'negative'
                processed_sentence.append(f"{word}[SENTIMENT={sentiment}]")
            else:
                processed_sentence.append(word)
        return processed_sentence


    grammar = """
    % start S
    S[NUM=sg, PERSON=3, SENTIMENT=positive] -> NP[NUM=sg, PERSON=3] VP[NUM=sg, PERSON=3, SENTIMENT=positive] | NP[NUM=sg, PERSON=3] Rel VP[NUM=sg, PERSON=3, SENTIMENT=positive] 
    S[NUM=sg, PERSON=3, SENTIMENT=negative] -> NP[NUM=sg, PERSON=3] VP[NUM=sg, PERSON=3, SENTIMENT=negative] | NP[NUM=sg, PERSON=3] VP[NUM=sg, PERSON=3, SENTIMENT=negative] CONJ NP[NUM=sg, PERSON=3] ADJ[SENTIMENT=negative] | PrepP ParP NP[NUM=sg, PERSON=3] VP[NUM=sg, PERSON=3, SENTIMENT=negative] | S[NUM=sg, PERSON=3, SENTIMENT=neutral] S[NUM=sg, PERSON=3, SENTIMENT=negative] |  NP[NUM=sg, PERSON=3, SENTIMENT=negative] VP[NUM=sg, PERSON=3] CONJ NP[NUM=sg, PERSON=3] VP[NUM=sg, PERSON=3, SENTIMENT=negative]
    S[NUM=sg, PERSON=3, SENTIMENT=neutral] -> NP[NUM=sg, PERSON=3]  VP[NUM=sg, PERSON=3] CONJ NP[NUM=pl, PERSON=3] PreP VP[NUM=pl, PERSON=3] 
    S[NUM=sg, PERSON=3, SENTIMENT=negative] -> NP[NUM=sg, PERSON=3] VP[NUM=sg, PERSON=3, SENTIMENT=positive] CONJ VP[NUM=sg, PERSON=3, SENTIMENT=negative] | NP[NUM=sg, PERSON=3] VP[NUM=sg, PERSON=3, SENTIMENT=positive] CONJ VP[NUM=sg, PERSON=3, SENTIMENT=negative]


    # Noun Phrase Structure
    NP[NUM=sg, PERSON=3] -> Det[NUM=sg] ADJ[NUM=sg] N[NUM=sg, PERSON=3] PrepP[NUM=sg] | Det[NUM=sg] N[NUM=sg, PERSON=3] PartP | ADJ[NUM=sg] N[NUM=sg, PERSON=3] CONJ N[NUM=sg, PERSON=3] | Det[NUM=sg] N[NUM=sg, PERSON=3] PrepP[NUM=sg] | Det[NUM=sg] N[NUM=sg, PERSON=3]
    NP[NUM=pl] -> Det[NUM=pl] N[NUM=pl]
    NP[NUM=sg, PERSON=3,SENTIMENT=negative] -> NEG ADJ[SENTIMENT=positive] N[NUM=sg, PERSON=3]
    NP[NUM=sg, PERSON=3] -> N[NUM=sg, PERSON=3] | N[NUM=sg, PERSON=3] NP[NUM=sg, PERSON=3] | Det[NUM=sg] N[NUM=sg, PERSON=3] PrepP[NUM=sg] PrepP[NUM=sg] 
    NP[NUM=pl, PERSON=3] -> N[NUM=sg] N[NUM=pl] | N[NUM=pl] 
    NP[NUM='sg', PERSON=3, SENTIMENT=negative] -> N[NUM=sg] ADJ[SENTIMENT=negative] | Det[NUM=sg] AdjP[SENTIMENT=negative] N[NUM=sg, SENTIMENT=negative]
    NP[NUM=sg, PERSON=3] -> Det[NUM=sg] N[NUM=sg, PERSON=3] CONJ N[NUM=sg, PERSON=3] PrepP | Det[NUM=sg] N[NUM=sg, PERSON=3] PrepP | P
    NP[NUM=sg, PERSON=3, SENTIMENT=positive] -> Det[NUM=sg] ADJ[SENTIMENT=positive] N[NUM=sg]

    # Verb Phrase Structure
    VP[NUM=sg, PERSON=3, SENTIMENT=negative] -> V[NUM=sg, PERSON=3] NEG ADJ[SENTIMENT=postitive] | V[NUM=sg, PERSON=3] ADJ[SENTIMENT=negative] | V[NUM=sg, PERSON=3] NEG V[NUM=sg, PERSON=3]
    VP[NUM=sg, PERSON=3, SENTIMENT=positive] -> V[NUM=sg, PERSON=3] NEG Det[NUM=sg] N[NUM=sg, PERSON=3, SENTIMENT=negative] | V[NUM=sg, PERSON=3] ADJ[SENTIMENT=positive] | V[NUM=sg, PERSON=3] NP[NUM=pl] |  V[NUM=sg, PERSON=3] Det[NUM=sg] N[NUM=sg, PERSON=3, SENTIMENT=positive] | V[NUM=sg, PERSON=3] NP[NUM=sg, PERSON=3, SENTIMENT=positive] 
    VP[NUM=sg, PERSON=3] -> V[NUM=sg, PERSON=3] PartP[NUM=sg] | V[NUM=sg, PERSON=3] N[NUM=sg, PERSON=3] | Adv V[NUM=sg, PERSON=3]
    VP[NUM=pl, PERSON=3] -> V[NUM=pl, PERSON=3] NP[NUM='sg', PERSON=3, SENTIMENT=negative]
    VP[NUM=pl, PERSON=3] -> V[NUM=pl, PERSON=3] CompP
    VP[NUM=sg, PERSON=3, SENTIMENT=negative] -> V[NUM=sg, PERSON=3] NP[NUM='sg', PERSON=3, SENTIMENT=negative] Inf

    # Prepositional Phrase Structure
    PreP -> P RelP Rel
    PrepP -> P NP[NUM=sg, PERSON=3] | P NP[NUM=pl, PERSON=3]
    PrepP[NUM=sg] -> P NP[NUM=sg, PERSON=3] | P N[NUM=sg, PERSON=3]

    PartP[NUM=sg] -> V NP[NUM=sg, PERSON=3] | V

    #parenthetical phrases
    ParP -> Adv NP[NUM=sg, PERSON=3]

    #Infinitive phrase
    Inf -> InfM V

    # Relative Clause Structure
    Rel -> Pronoun VP[NUM=sg, PERSON=3, SENTIMENT=positive] | NP[NUM=pl, PERSON=3] VP[NUM=pl, PERSON=3] 

    CompP -> AdjP NP[NUM=pl, PERSON=3]

    #Adjective phrase
    AdjP -> Det Adj P
    AdjP[SENTIMENT=negative] -> Adv ADJ[SENTIMENT=negative]

    N[NUM=sg, PERSON=3] -> 'John' | 'Carpenter' | 'action' | 'horror' | 'writer' | 'director' | 'expert' | 'production' | 'script' | 'film' | 'acting' | 'originality[SENTIMENT=positive]' | 'performance' | 'Election' | 'director' | 'dialogue' | 'pacing' | 'acting' | 'Broderick'
    N[NUM=pl] ->  'scenes' | 'people' | 'films'
    N[NUM=sg] -> 'something' | 'ending'
    N[NUM=sg, SENTIMENT=negative] -> 'mistake[SENTIMENT=negative]' | 'failure'
    N[NUM=sg, PERSON=3, SENTIMENT=positive] -> 'masterpiece[SENTIMENT=positive]'

    V[NUM=sg, PERSON=3] -> 'believes' | 'is' | 'creates' | 'won[SENTIMENT=positive]' | 'was' | 'lacks[SENTIMENT=negative]' | 'has' | 'does' | 'entertain[SENTIMENT=positive]'
    V[NUM=pl, PERSON=3] -> 'fight' | 'are'
    V -> 'make' | 'lacking[SENTIMENT=negative]' | 'involved' 

    ADJ[SENTIMENT=negative] -> 'horrible[SENTIMENT=negative]' | 'bad[SENTIMENT=negative]' | 'tedious[SENTIMENT=negative]' | 'slow[SENTIMENT=negative]'
    Adj -> 'same' | 'bad'
    ADJ[SENTIMENT=positive] -> 'brilliant[SENTIMENT=positive]' | 'fantastic[SENTIMENT=positive]' | 'real' | 'good[SENTIMENT=positive]'
    Adv -> 'apparently' | 'supposedly' | 'very'

    Det -> 'the' 
    Det[NUM=sg] -> 'a' | 'an' | 'The' | 'the' | 'a' | 'No'
    Det[NUM=pl] -> 'several'

    CONJ -> 'that' | 'and' | 'but'

    P -> 'in' | 'as' | 'For' | 'of' | 'on' | 'it' | 'for'

    RelP -> 'which'

    NEG -> 'not' | 'no'

    InfM -> 'to'

    Pronoun -> 'who'

    """
    ent = ""
    for e in entity:
            ent = ent  +e+ " "

    feature_grammar = FeatureGrammar.fromstring(grammar)
    parser = FeatureEarleyChartParser(feature_grammar)
    
    for sentence in sentences:
        sentence = sentence.split()
        preprocessed_sentence = preprocess_sentence(sentence)

        
        
        trees = list(parser.parse(preprocessed_sentence))
        for tree in trees: 
            print("---------------------")
            print("\n")
            print(tree)
            print("\n")
            sentiment = extract_entity_sentiment(tree, entity[0])
            overall_sentiment = extract_overall_sentiment(tree)
            print("Overall sentiment of the sentence:",overall_sentiment)
            print("\n")
            print("Stance towards entity:" + ent +"= "+ sentiment)
            print("\n")

            break
