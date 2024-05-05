import nltk
'''
    In order to better integrate the Gazetteer for this example, i have replaced the names of the keys in the gazetteer list with
    the categories in the NER module. This can be used to better train the Named entity recognition in order to give better results
'''
import nltk

def custom_integration(ne_tree, gazetteer_annotation):
    refined_ne_tree = []
    for i, subtree in enumerate(ne_tree):
        if isinstance(subtree, nltk.Tree): 
            label = subtree.label() 
            leaves = subtree.leaves()
        
            for j, word in enumerate(leaves):  
                annotated = gazetteer_annotation[i + j]  # get the gazetteer annotation for the word
                # Modify the label if the word has a specific annotation in gazetteer
                if 'country' in annotated:  
                    label = 'GPE'  # change the label to GPE
                elif 'currency' in annotated:
                    label = 'MONEY'
                elif 'unit' in annotated:
                    label = 'UNIT'
                    
            refined_subtree = nltk.Tree(label, leaves)  # create a new subtree with the modified label
            refined_ne_tree.append(refined_subtree)  # append the modified subtree
        else:  
            refined_ne_tree.append(subtree)  # append the subtree (word, tag) as is
            
    return nltk.Tree('S', refined_ne_tree)  # return a new tree with all the (modified) subtrees

def perform_ner(tags, gazetteer_annotation):
    ne_tree = nltk.ne_chunk(tags, binary=False) 
    return custom_integration(ne_tree, gazetteer_annotation)

