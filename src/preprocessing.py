import nltk
from nltk.corpus import stopwords 

nltk.download('stopwords')   
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

import string

ps = PorterStemmer()

# Limpa o texto para ser usado pelos modelos
def transformar_texto(texto):
    # 1) Deixa tudo em minúsculas
    texto = texto.lower()
    
    # 2) Quebra o texto em palavras (tokens)
    texto = nltk.word_tokenize(texto)
    
    # 3) Mantém só letras e números
    tokens = []
    for palavra in texto:
        if palavra.isalnum():
            tokens.append(palavra)
            
    texto = tokens[:]
    tokens.clear()
    
    # 4) Remove stopwords ("the", "is"...) e pontuação
    for palavra in texto:
        if palavra not in stopwords.words('english') and palavra not in string.punctuation:
            tokens.append(palavra)
        
    # 5) Reduz cada palavra ao radical
    texto = tokens[:]
    tokens.clear()
    for palavra in texto:
        tokens.append(ps.stem(palavra))
    
    # 6) Junta as palavras de volta em uma string
    return " ".join(tokens)