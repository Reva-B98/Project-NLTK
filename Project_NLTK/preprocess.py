import sys
import nltk
from nltk.corpus import reuters
from nltk.tree import Tree

import tokenization
import gazetteer_annotation
import named_entity_recognition
import measured_entity_recognition
import feature_grammar

def PreProcess(reuter_text):
    raw_text = reuter_text
    print(raw_text)
    print("\n")
    
    print("1. TOKENIZATION")
    tokens = tokenization.tokenize_text(raw_text)
    print(tokens)
    print("\n")
    
    print("2. SENTENCE SPLITTING")
    print("\n")
    processed_text = tokenization.sentence_splitting(raw_text)
    for sent in processed_text:
        print(sent)
        print("\n")
    
    print("3. POS TAGGING")
    print("\n")
    tags = nltk.pos_tag(tokens)
    print(tags)
    print("\n")

    print("4. GAZETTEER ANNOTATION")
    print("\n")
    gazetteers = {
        'country': {
            'Canada', 'Denmark', 'England', 'France', 'Germany', 'Hungary', 'Indonesia',
            'USA', 'India', 'China', 'Russia', 'Australia', 'Brazil', 'South Africa', 'Japan',
            'Mexico', 'Philippines', 'Singapore', 'New Zealand', 'Pakistan', 'Nigeria', 'Bangladesh',
            'Turkey', 'Norway', 'Sweden', 'Finland', 'Iceland', 'Argentina', 'Chile', 'Peru'
        },
        'currency': {
            'USD', 'EUR', 'GBP', 'IDR', 'INR', 'CNY', 'RUB', 'AUD', 'BRL', 'ZAR', 'JPY', 'MXN', 'PHP', 
            'SGD', 'NZD', 'PKR', 'NGN', 'BDT', 'TRY', 'NOK', 'SEK', 'ISK', 'ARS', 'CLP', 'PEN'
        }, 'unit': {
            'tonnes', 'kg', 'meters', 'km', 'cm', 'mm', 'miles', 'yards', 'feet', 'inches', 'liters', 
            'gallons', 'quarts', 'pints', 'cups', 'fluid ounces', 'tablespoons', 'teaspoons', 'grams', 'milligrams', 'micrograms'
        }
        
    }
    gazetteer_annotations = gazetteer_annotation.annotate_gazetteer(tokens, gazetteers)
    print(gazetteer_annotations)
    print("\n")

    print("5. NAMED ENTITY RECOGNITION")
    print("\n")
    refined_ne_tree = named_entity_recognition.perform_ner(tags, gazetteer_annotations)
    print(refined_ne_tree)
    print("\n")

    print("6. MEASURED ENTITY RECOGNITION")
    print("\n")
    units_gazetteer = {
        'unit': {
            'tonnes', 'kg', 'meters', 'km', 'cm', 'mm', 'miles', 'yards', 'feet', 'inches', 'liters', 
            'gallons', 'quarts', 'pints', 'cups', 'fluid ounces', 'tablespoons', 'teaspoons', 'grams', 'milligrams', 'micrograms'
        },
        'date': {
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
        'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Today', 'Yesterday', 'Tomorrow', 'Days', 'Weeks', 'Months', 'Years'
        },
        'time': {
            'AM', 'PM', 'am', 'pm', 'Hour', 'Minute', 'Second', 'Hr', 'Min', 'Sec', 'Hours', 'Minutes', 'Seconds', 'Hrs', 'Mins', 'Secs'
        }
    }
    unit_gazetteer = {unit.lower() for unit in units_gazetteer['unit']}
    date_gazetteer = {date.lower() for date in units_gazetteer['date']}
    time_gazetteer = {time.lower() for time in units_gazetteer['time']}
    measured_entities = measured_entity_recognition.measured_entity(tokens, unit_gazetteer, date_gazetteer, time_gazetteer)
    print(measured_entities)
    print("\n")
    print("7.Feature Grammar")
    
    def extract_person_entities(ne_tree):
        person_entities = []

        for subtree in ne_tree:
            if isinstance(subtree, Tree) and subtree.label() in ('PERSON','GPE'):
                person_name = " ".join([token for token, pos in subtree.leaves()])
                person_entities.append(person_name)

        return person_entities
    entity = extract_person_entities(refined_ne_tree)
    feature_grammar.featuregrammar(sentences,entity)


if __name__ == "__main__":
    sentences = [
   "John Carpenter apparently believes that action scenes in which people fight something horrible are the same as horror scenes. For a writer and director of horror films , supposedly an expert on horror , it is a very bad mistake to make"
    ]   
    reuter_text = sentences[0]
    PreProcess(reuter_text)
