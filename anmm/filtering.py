import re
import unicodedata
from num2words import num2words
import math

import pandas as pd

dict_file="acronym_dict.csv"

def save_acronym_dict(dict,path):
     df = pd.DataFrame.from_dict(dict, orient="index")
     df.to_csv(path)
     return 

def load_acronym_dict(path):
    df = pd.read_csv(path, index_col=0)
    d = df.to_dict("split")
    d = (zip(d["index"], d["data"]))
    dict={}
    for i,a in enumerate(d):  
        dict[a[0]]=a[1]
    return dict
acronym_dict=load_acronym_dict(dict_file)
acronym_dict

#convert string into unicode
def unicodeToAscii(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

#  if string's a string of numbers including float and int
# “32”.is_number=true;  "31.23".isnumber=True

acronym_dict=load_acronym_dict(dict_file)

def is_number(text):
    try:
        a=float(text)
        if math.isnan(a):
            return False 
        return True
    except ValueError:
        return False
    
    
def speical_char_processing(text):
    text= re.sub(r"&", r"and", text).strip()
    text= re.sub(r"%", r" percent", text)
    text= re.sub(r"e\.g\.", r"such as", text)
    text= re.sub(r"\$", r" dollar ", text)
    return text

def acronym_replace(text,dict):
    
    t=re.findall(r'\b[A-Z]{1,}[a-z]*[A-Z]{1,}\b',text)
    t=list(set(t))
    for i in range (len(t)):
        if(acronym_dict.get(t[i])!=None):
             text=text.replace(t[i],dict.get(t[i])[0])
    return text
    
def trim_punctuation(text):
    text= re.sub(r"[^0-9a-zA-Z.!?$]+", r" ", text).strip()
    
    #processing in-sentences puncutations, segmenting each sentences using "."
    text=re.sub(r'([^\w]*)[?|!]','.',text)
    text=re.sub(r'\.{2,}',r'.',text)
    text=re.sub(r'(\. ){2,}',r'.',text)
    text=re.sub(r'(\.){2,}',r'.',text)
    text=re.sub(r' \. ',r'.',text)
    text=re.sub(r' \.',r'.',text)
    text=re.sub(r'\.',r'. ',text)
    
    #process front side and end 
    text=re.sub(r'^(\.)',r'',text.strip())
    if text!='' and re.findall(r'.$',text)[0]!='.':
        text=re.sub(r'(\w*$)',r'\1.',text)
    return text
def conert_number2string(text):
    
    t=text.split(" ")
    c=[ num2words((float)(a)) if is_number(a) else a for a in t ]
    return " ".join(c)

def trim_clipped_word(text):
    text = re.sub(r"\bi m \b",r"i am ",text)
    text = re.sub(r"\bit s \b",r"it is ",text)
    text = re.sub(r"\bhe s \b",r"he is ",text)
    text = re.sub(r"he s been \b",r"he has ",text)
    text = re.sub(r"\bwon t \b",r"will not ",text)
    text = re.sub(r"\bn t \b",r"n can not ",text)
    text = re.sub(r"\b ll \b",r" will ",text)
    text = re.sub(r"\b re \b",r" are ",text)
    text = re.sub(r"\b ve \b",r"they have ",text)
    text = re.sub(r"\b d been \b",r" had been ",text)
    text = re.sub(r"\b d \b",r" would ",text)
    text = re.sub(r"\b s \b",r" ",text)
    text=text.strip()
    return text

#input text is a concat of "Idea Title","Idea Description","Problem","Solution"
def normalizeDocument(text):
    text=unicodeToAscii(text)
    text=speical_char_processing(text)
    
    ##dolloar,email,phone processing,number
    text=conert_number2string(text)
    text=trim_punctuation(text)
    text=conert_number2string(text)
    text=acronym_replace(text,acronym_dict) 
    text=trim_clipped_word(text)             
    return text
    