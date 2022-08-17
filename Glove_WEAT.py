import numpy as np
from utils import mean_cosine_sim, trans_matrix, stdv
import statistics
import math
import random
from scipy.stats import norm, gaussian_kde
import itertools

# WEAT 1 and 2
flowers = ["aster", "clover", "hyacinth", "marigold", "poppy",
        "azalea", "crocus", "iris", "orchid", "rose", "bluebell",
        "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy",
        "lily", "peony", "violet", "carnation", "gladiola",
        "magnolia", "petunia", "zinnia"]

insects = ["ant", "caterpillar", "flea", "locust", "spider", "bedbug",
        "centipede", "fly", "maggot", "tarantula",
        "bee", "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket",
        "hornet", "moth", "wasp", "blackfly",
        "dragonfly", "horsefly", "roach", "weevil"]

pleasant = ["caress", "freedom", "health", "love", "peace", "cheer",
        "friend", "heaven", "loyal", "pleasure", "diamond", "gentle", "honest",
        "lucky", "rainbow", "diploma", "gift", "honor", "miracle",
        "sunrise", "family", "happy", "laughter", "paradise", "vacation"]

unpleasant = ["abuse", "crash", "filth", "murder", "sickness", "accident",
            "death", "grief", "poison", "stink", "assault", "disaster",
            "hatred", "pollute", "tragedy", "divorce", "jail", "poverty",
            "ugly", "cancer", "kill", "rotten", "vomit", "agony", "prison"]

instruments = ["bagpipe", "cello", "guitar", "lute", "trombone", "banjo", 
            "clarinet", "harmonica", "mandolin", "trumpet", "bassoon", "drum",
            "harp", "oboe", "tuba", "bell", "fiddle", "harpsichord", "piano", 
            "viola", "bongo", "flute", "horn", "saxophone", "violin"]

weapons = ["arrow", "club", "gun", "missile", "spear", "axe", "dagger", 
        "harpoon", "pistol", "sword", "blade", "dynamite", "hatchet", 
        "rifle", "tank", "bomb", "firearm", "knife", "shotgun", "teargas", 
        "cannon", "grenade", "mace", "slingshot", "whip"]

# WEAT 3
euro_3 = ["Adam", "Harry", "Josh", "Roger", "Alan", "Frank", "Justin", "Ryan", "Andrew", "Jack", 
        "Matthew", "Stephen", "Brad", "Greg", "Paul", "Jonathan", "Peter", "Amanda", "Courtney", 
        "Heather", "Melanie", "Katie", "Betsy", "Kristin", "Nancy", "Stephanie",
        "Ellen", "Lauren", "Colleen", "Emily", "Megan", "Rachel"]
african_3 = ["Alonzo", "Jamel", "Theo", "Alphonse", "Jerome", "Leroy", "Torrance", "Darnell", "Lamar", "Lionel",
        "Tyree", "Deion", "Lamont", "Malik", "Terrence", "Tyrone", "Lavon", "Marcellus", "Wardell", "Nichelle",
        "Shereen", "Ebony", "Latisha", "Shaniqua", "Jasmine", "Tanisha", "Tia", "Lakisha", "Latoya", "Yolanda",
        "Malika", "Yvette"]

pleasant_3 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal",
            "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift",
            "honor", "miracle", "sunrise", "family", "happy", "laughter", "paradise", "vacation"]

unpleasant_3 = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", 
                "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy",
                "bomb", "divorce", "jail", "poverty", "ugly", "cancer", "evil",
                "kill", "rotten", "vomit"]

# WEAT 4
euro_4 = ["Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Matthew", "Neil",
        "Todd", "Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie", "Meredith", "Sarah"]
    
african_4 = ["Darnell", "Hakim", "Jermaine", "Kareem", "Jamal", "Leroy", "Rasheed", "Tyrone",
            "Aisha", "Ebony", "Keisha", "Kenya", "Lakisha", "Latoya", "Tamika", "Tanisha"]

pleasant_4 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven",
            "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma",
            "gift", "honor", "miracle", "sunrise", "family",
            "happy", "laughter", "paradise", "vacation"]

unpleasant_4 = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death",
                "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute",
                "tragedy", "bomb", "divorce", "jail", "poverty", "ugly", "cancer", "evil",
                "kill", "rotten", "vomit"]

# WEAT 5
euro_5 = ["Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Matthew", "Neil",
        "Todd", "Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie", "Meredith", "Sarah"]

african_5 = ["Darnell", "Hakim", "Jermaine", "Kareem", "Jamal", "Leroy", "Rasheed", "Tyrone",
            "Aisha", "Ebony", "Keisha", "Kenya", "Lakisha", "Latoya", "Tamika", "Tanisha"]

pleasant_5 = ["joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy"]

unpleasant_5 = ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"]

# WEAT 6
male_names = ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"]
female_names = ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"]
career = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
family = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]

# WEAT 7
math_7 = ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"]
arts_7 = ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"]
male_7 = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
female_7 = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

# WEAT 8
science = ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"]
arts = ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama"]
male_terms = ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"]
female_terms = ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "her"]

# WEAT 9
mental_disease = ["sad", "hopeless", "gloomy", "tearful", "miserable", "depressed"]
physical_disease = ["sick", "illness", "influenza", "disease", "virus", "cancer"]
temporary = ["impermanent", "unstable", "variable", "fleeting", "short", "brief", "occasional"]
permanent = ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"]

# WEAT 10
young_names = ["Tiffany", "Michelle", "Cindy", "Kristy", "Brad", "Eric", "Joey", "Billy"]
old_names = ["Ethel", "Bernice", "Gertrude", "Agnes", "Cecil", "Wilbert", "Mortimer", "Edgar"]
pleasant_10 = ["joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy"]
unpleasant_10 = ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"]

# 'glove.WEAT.300d.txt' is the file that only contains the words we need in WEAT 1-10 from the glove.840B.300d.txt.
model = {}
with open('glove.WEAT.300d.txt', encoding='utf-8') as e:
    lines = e.read().splitlines()
    for line in lines:
        list_emb = line.split(" ")
        model[list_emb[0]] = np.array([float(list_emb[i]) for i in range(1, len(list_emb))])
    e.close()


def test_stat(X, Y, A_matrix, B_matrix):
    '''
    Returns the s(X, Y, A, B) score. 
    mean_cosine_sim computes the half of s(w, A, B)
    '''
    sum_x = 0
    sum_y = 0
    for x in X:
        sum_x += mean_cosine_sim(A_matrix, model[x]) - mean_cosine_sim(B_matrix, model[x])
    for y in Y:
        sum_y += mean_cosine_sim(A_matrix, model[y]) - mean_cosine_sim(B_matrix, model[y])
    return sum_x - sum_y

def effect_size(X, Y, A, B):
    X_list = []
    Y_list = []
    for x in X:
        X_list.append(mean_cosine_sim(A, model[x]) - mean_cosine_sim(B, model[x]))
    #print(model[x].shape)
    for y in Y:
        Y_list.append(mean_cosine_sim(A, model[y]) - mean_cosine_sim(B, model[y]))
    X_and_Y = X_list + Y_list
    nom = statistics.mean(X_list) - statistics.mean(Y_list) 
    dev = statistics.stdev(X_and_Y)
    return round(nom / dev, 2)
  

def p_value(X, Y, A, B):
    '''
    Calculate P values of WEAT 1-5
    '''
    all_results = []    # Stores the 100000 points
    # Random select 100000 lengths of the Xi, Yi samples
    num_comb = []
    for i in range(1, len(X)+1):
      num_comb.append(math.comb(len(X), i)**2)    # An array of cumulative number of combinations for a given length 
    nums = random.choices(list(range(1, len(X)+1)), k=100000, weights=num_comb) # Generate 100000 random samples
    
    # Stores all embeddings of the attribute word in A and B into a big matrix
    A_matrix = trans_matrix(A, model)
    B_matrix = trans_matrix(B, model)
    visited = set() # Stores all seen samples
    overall_test_stats = test_stat(X, Y, A_matrix, B_matrix)    # The overall test statistics we need to compare with
    for count in nums:
      x = frozenset(random.sample(X, k=count))
      y = frozenset(random.sample(Y, k=count))
      while x.union(y) in visited:  # Resamples when the chosen sample is seen
        x = frozenset(random.sample(X, k=count))
        y = frozenset(random.sample(Y, k=count))
      visited.add(x.union(y))
      all_results.append(test_stat(x, y, A_matrix, B_matrix))
    return cal_p_value(overall_test_stats, all_results)

def p_value_610(X, Y, A, B):
    '''
    Calculate P values of WEAT 6-10
    '''
    all_results = []
    A_matrix = trans_matrix(A, model)
    B_matrix = trans_matrix(B, model)
    for l in range(len(X)):
        combs_X = list(itertools.combinations(X, l+1))
        combs_Y = list(itertools.combinations(Y, l+1))
        for Xi in combs_X:
            for Yi in combs_Y:
                all_results.append(test_stat(Xi, Yi, A_matrix, B_matrix))
    overall_results = test_stat(X, Y, A_matrix, B_matrix)
    return cal_p_value(overall_results, all_results)

def cal_p_value(x, results):
    KDE = gaussian_kde(results) # Use a gaussian KDE to estimate the density of 100000 random samples
    mean = sum(results) / len(results)
    std = stdv(results)   # Uses a normal distribution to estimate the density

    # Find the 1-cdf of kde and p value of the normal distribution
    return KDE.integrate_box_1d(x, np.inf), norm.sf(x, loc=mean, scale=std), 1-norm.cdf(x, loc=mean, scale=std)

'''
print(test_stat(flowers, insects, trans_matrix(pleasant, model), trans_matrix(unpleasant, model)))    # WEAT 1
print(test_stat(instruments, weapons, trans_matrix(pleasant, model), trans_matrix(unpleasant, model)))    # WEAT 2
print(test_stat(euro_3, african_3, trans_matrix(pleasant_3, model), trans_matrix(unpleasant_3, model)))   # WEAT 3
print(test_stat(euro_4, african_4, trans_matrix(pleasant_4, model), trans_matrix(unpleasant_4, model)))   # WEAT 4
print(test_stat(euro_5, african_5, trans_matrix(pleasant_5, model), trans_matrix(unpleasant_5, model)))    # WEAT 5
print(test_stat(male_names, female_names, trans_matrix(career, model), trans_matrix(family, model)))  # WEAT 6
print(test_stat(math_7, arts_7, trans_matrix(male_7, model), trans_matrix(female_7, model)))   # WEAT 7
print(test_stat(science, arts, trans_matrix(male_terms, model), trans_matrix(female_terms, model)))   # WEAT 8
print(test_stat(mental_disease, physical_disease, trans_matrix(temporary, model), trans_matrix(permanent, model)))    # WEAT 9
print(test_stat(young_names, old_names, trans_matrix(pleasant_10, model), trans_matrix(unpleasant_10, model)))    # WEAT 10
print("------------")
'''
print(effect_size(flowers, insects, trans_matrix(pleasant, model), trans_matrix(unpleasant, model)))    # WEAT 1
print(effect_size(instruments, weapons, trans_matrix(pleasant, model), trans_matrix(unpleasant, model)))    # WEAT 2
print(effect_size(euro_3, african_3, trans_matrix(pleasant_3, model), trans_matrix(unpleasant_3, model)))   # WEAT 3
print(effect_size(euro_4, african_4, trans_matrix(pleasant_4, model), trans_matrix(unpleasant_4, model)))   # WEAT 4
print(effect_size(euro_5, african_5, trans_matrix(pleasant_5, model), trans_matrix(unpleasant_5, model)))    # WEAT 5
print(effect_size(male_names, female_names, trans_matrix(career, model), trans_matrix(family, model)))  # WEAT 6
print(effect_size(math_7, arts_7, trans_matrix(male_7, model), trans_matrix(female_7, model)))   # WEAT 7
print(effect_size(science, arts, trans_matrix(male_terms, model), trans_matrix(female_terms, model)))   # WEAT 8
print(effect_size(mental_disease, physical_disease, trans_matrix(temporary, model), trans_matrix(permanent, model)))    # WEAT 9
print(effect_size(young_names, old_names, trans_matrix(pleasant_10, model), trans_matrix(unpleasant_10, model)))    # WEAT 10
print("------------")
'''
print(p_value(flowers, insects, pleasant, unpleasant))  # WEAT 1
print(p_value(instruments, weapons, pleasant, unpleasant))  # WEAT 2
print(p_value(euro_3, african_3, pleasant_3, unpleasant_3))    # WEAT 3
print(p_value(euro_4, african_4, pleasant_4, unpleasant_4))    # WEAT 4
print(p_value(euro_5, african_5, pleasant_5, unpleasant_5))    # WEAT 5
print(p_value_610(male_names, female_names, career, family))  # WEAT 6
print(p_value_610(math_7, arts_7, male_7, female_7))    # WEAT 7
print(p_value_610(science, arts, male_terms, female_terms)) # WEAT 8
print(p_value_610(mental_disease, physical_disease,temporary, permanent))  # WEAT 9
print(p_value_610(young_names, old_names, pleasant_10, unpleasant_10))  # WEAT 10

print(effect_size(flowers, insects, trans_matrix(pleasant, model), trans_matrix(unpleasant, model)))    # WEAT 1
print(effect_size(instruments, weapons, trans_matrix(pleasant, model), trans_matrix(unpleasant, model)))    # WEAT 2
print(effect_size(euro_3, african_3, trans_matrix(pleasant_3, model), trans_matrix(unpleasant_3, model)))   # WEAT 3
print(effect_size(euro_4, african_4, trans_matrix(pleasant_4, model), trans_matrix(unpleasant_4, model)))   # WEAT 4
print(effect_size(euro_5, african_5, trans_matrix(pleasant_5, model), trans_matrix(unpleasant_5, model)))    # WEAT 5
print(effect_size(male_names, female_names, trans_matrix(career, model), trans_matrix(family, model)))  # WEAT 6
print(effect_size(math_7, arts_7, trans_matrix(male_7, model), trans_matrix(female_7, model)))   # WEAT 7
print(effect_size(science, arts, trans_matrix(male_terms, model), trans_matrix(female_terms, model)))   # WEAT 8
print(effect_size(mental_disease, physical_disease, trans_matrix(temporary, model), trans_matrix(permanent, model)))    # WEAT 9
print(effect_size(young_names, old_names, trans_matrix(pleasant_10, model), trans_matrix(unpleasant_10, model)))    # WEAT 10
'''