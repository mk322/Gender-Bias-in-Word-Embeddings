import statistics
import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from utils import mean_cosine_sim, stdv, trans_matrix, std_cosine_sim, angle_between

# WEAT 1 and 2
flowers = ["aster", "clover", "hyacinth", "marigold", "poppy",
        "azalea", "crocus", "iris", "orchid", "rose", "bluebell",
        "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy",
        "lily", "peony", "violet", "carnation", "gladiolas",
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

weapons = ["arrow", "club", "gun", "missile", "spear", "axes", "dagger", 
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

weaft_female = ["sister" , "female" , "woman" , "girl" , "daughter" , "she" , "hers" , "her"]
weaft_male = ["brother" , "male" , "man" , "boy" , "son" , "he" , "his" , "him"]

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

model = {}
with open('glove.WEFAT.300d.txt', encoding='utf-8') as e:
    lines = e.read().splitlines()
    for line in lines:
        list_emb = line.split(" ")
        model[list_emb[0]] = np.array([float(list_emb[i]) for i in range(1, len(list_emb))])
    e.close()

print(len(model["kang"]))

def WEFAT_score(x, A_matrix, B_matrix, model):
    nom = mean_cosine_sim(A_matrix, model[x]) - mean_cosine_sim(B_matrix, model[x])
    AB_matrix = np.concatenate((A_matrix, B_matrix))
    denom = std_cosine_sim(AB_matrix, model[x])
    return nom/denom

def cos_sim(A, B):
        return np.dot(A,B)/(norm(A)*norm(B))

def cosine_distance(a, b):
     if a.shape != b.shape:
         raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
     if a.ndim==1:
         a_norm = np.linalg.norm(a)
         b_norm = np.linalg.norm(b)
     elif a.ndim==2:
         a_norm = np.linalg.norm(a, axis=1, keepdims=True)
         b_norm = np.linalg.norm(b, axis=1, keepdims=True)
     else:
         raise RuntimeError("array dimensions {} not right".format(a.ndim))
     similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
     dist = 1. - similiarity
     return dist

def WEFAT_s(x, A, B, model):
        A_list = []
        B_list = []
        A_and_B = []
        for a in A:
                A_list.append(cosine_distance(model[a], model[x]))
        for b in B:
                B_list.append(cosine_distance(model[b], model[x]))
        nom = statistics.mean(A_list) - statistics.mean(B_list)
        denom = stdv(A_list + B_list)
        print(nom)
        print(denom)
        return nom/denom

print("WEFAT Scores of murray: ", WEFAT_s("murray", weaft_female, weaft_male, model))
print("WEFAT Scores of Murray: ", WEFAT_s("Murray", weaft_female, weaft_male, model))
print("WEFAT Scores of kang: ", WEFAT_s("kang", weaft_female, weaft_male, model))
print("WEFAT Scores of Kang: ", WEFAT_s("Kang", weaft_female, weaft_male, model))


print("WEFAT Scores of murray: ", WEFAT_score("murray", trans_matrix(weaft_female, model), trans_matrix(weaft_male, model), model))
print("WEFAT Scores of Murray: ", WEFAT_score("Murray", trans_matrix(weaft_female, model), trans_matrix(weaft_male, model), model))
print("WEFAT Scores of Kang: ", WEFAT_score("Kang", trans_matrix(weaft_female, model), trans_matrix(weaft_male, model), model))
print("WEFAT Scores of kang: ", WEFAT_score("kang", trans_matrix(weaft_female, model), trans_matrix(weaft_male, model), model))
'''
print("WEFAT Scores of murray: ", WEFAT_score("murray", trans_matrix(math_7, model), trans_matrix(arts_7, model), model))
print("WEFAT Scores of Murray: ", WEFAT_score("Murray", trans_matrix(math_7, model), trans_matrix(arts_7, model), model))
print("WEFAT Scores of Kang: ", WEFAT_score("Kang", trans_matrix(math_7, model), trans_matrix(arts_7, model), model))
print("WEFAT Scores of kang: ", WEFAT_score("kang", trans_matrix(math_7, model), trans_matrix(arts_7, model), model))
'''