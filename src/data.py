import pandas as pd
from sklearn.preprocessing import LabelEncoder


def carregar_dados(caminho="data/raw/spam.csv"):
    df = pd.read_csv(caminho, encoding="latin-1")
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])  # ham=0, spam=1

    df = df.drop_duplicates(keep='first')
    return df