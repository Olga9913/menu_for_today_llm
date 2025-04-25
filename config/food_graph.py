from collections import defaultdict
import spacy
from tqdm import tqdm
import re
import networkx as nx
import pickle

from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
import nltk

from fuzzywuzzy import process


nltk.download('stopwords')

nlp = spacy.load("ru_core_news_sm")

ATTRIBUTES_ORDER = [
    'mainIngridients',
    'ingridients',
    'diet',
    'meal',
    'occasions',
    'geography',
]

PATTERNS = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(PATTERNS, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            
            tokens.append(token)
    return tokens

def make_tags_list(recipes_list, min_count=10):
    
    tags = defaultdict(int)
    
    for attribute in ATTRIBUTES_ORDER:
        for recipe in recipes_list:
            tags_list = getattr(recipe, attribute)
            if tags_list is not None:
                for tag in tags_list:
                    if attribute == 'ingridients':
                        tags[tag[0].lower()] += 1
                    else:
                        tags[tag.lower()] += 1
            
    tags_list = list(tags.keys()) 
    for tag in tags_list:
        if tags[tag] < min_count:
            del tags[tag]
        
    return tags


def make_one_word_tags_list(tags):
    res = {}
    for key in tags.keys():
        if len(key.split(" ")) == 1:
           res[key] = tags[key]
    return res


LEVENSTEIN_SIMILARITY_MIN = 0.5
def levenstein_similarity_normalized(text1: str, text2: str) -> float:
    """
    Compute the normalized levenstein distance between two texts.
    """
    import nltk
    return 1 - nltk.edit_distance(text1, text2) / max(len(text1), len(text2))
    

def enreach_query_with_relative_tags(query, one_word_tags):
    query = re.sub(PATTERNS, ' ', query.lower())
    new_tags = []
    for token in query.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            close_token = process.extractOne(token, [key for key in one_word_tags.keys()])
            if levenstein_similarity_normalized(token, close_token[0]) >= LEVENSTEIN_SIMILARITY_MIN:
                new_tags.append(close_token[0])
    return new_tags

        

def lemmatize_sentance(text):
    return lemmatize(text)

    

def lemmatize_tags(tags):
    nlp = spacy.load("ru_core_news_sm")
    for tag in tags:
        if "(" in tag:
            new_tag = re.sub(r'\([^)]*\)', '', tag).strip()
        else:
            new_tag = tag
        tags[tag] = {
            'stat': tags[tag],
            'lemma': tuple(lemmatize_sentance(new_tag))
        }   



def add_recipes_to_graph(G, recipes_list):
    for recipe in recipes_list:
        G.add_node(recipe.id, node_type="recipe")
        G.nodes[recipe.id]['name'] = recipe.name
    

def add_tags_to_graph(G, tags):
    for tag in tags:
        G.add_node(tags[tag]['lemma'], node_type="tag")


def add_edges_to_graph(G, recipes_list, tags):
    for attribute in ATTRIBUTES_ORDER:
        for recipe in recipes_list:
            recipe_tags_list = getattr(recipe, attribute)
            if recipe_tags_list is not None:
                for tag in recipe_tags_list:
                    if attribute != 'ingridients':
                        if tag in tags: 
                            if G.has_node(tags[tag]['lemma']):
                                G.add_edge(recipe.id, tags[tag]['lemma'])
                            else:
                                print(f"Node {tags[tag]['lemma']} does not exist")
                    else:
                        if tag[0] in tags: 
                            if G.has_node(tags[tag[0]]['lemma']):
                                G.add_edge(recipe.id, tags[tag[0]]['lemma'])
                            else:
                                print(f"Node {tags[tag]['lemma']} does not exist")


def build_knowledge_graph(recipes_list, tags):
    
    G = nx.Graph()
    add_recipes_to_graph(G, recipes_list)
    add_tags_to_graph(G, tags)
    add_edges_to_graph(G, recipes_list, tags)
    
    return G
    


def query_graph(query, graph, tags, min_number=5, verbose=False):
    query_lemms = lemmatize_sentance(query)
    
    
    answer = set()
    answer_new = set()
    
    for n in graph.nodes():
        if graph.nodes[n]['node_type'] == 'recipe':
            answer.add(n)
    for tag in tags:
        if len(set(query_lemms).intersection(set(tags[tag]['lemma'])))\
                == len(set(tags[tag]['lemma'])):
            current_subgraph = set()
            for n in graph.neighbors(tags[tag]['lemma']):
                current_subgraph.add(n)
            answer_new = answer.intersection(current_subgraph)
            if verbose:
                print(f" Tag ({tag, tags[tag]}) is applied. Selected {len(answer_new)} recipes")
            if len(answer_new) < min_number:
                return answer_new, answer
            answer = answer_new
    return answer_new, answer

def save_graph(graph, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)