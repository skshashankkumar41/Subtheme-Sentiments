import pandas as pd 
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

# loading the dataframe and doing basic preprocessing
def loader(dfPath):
    df = pd.read_csv(dfPath,header = None)
    df = df.fillna('')
    
    # to get all the multi labels in one column
    columns = ['text']
    labels = []
    
    for idx in range(1, 15):
        name = 'label_' + str(idx)
        labels.append(name)
        columns.append(name)

    df.columns = columns

    df['target'] = df[labels].values.tolist()

    df =  df[['text','target']]

    return df 

# remove null labels 
def remove_empty(text):
    return [lab for lab in text if lab != '']

# remove extra spaces from labels
def remove_space(text):
    return [lab.strip() for lab in text]

# to replace duplicate labels with correct labels
def replace_label(df, src, trg):
    def replace(texts):
        return [lab if lab != src else trg for lab in texts]
    
    df['target'] = df['target'].map(replace)

# to get all noisy labels that don't have any sentiments 
def get_noisy_labels(df):
    noisy_labels = []
    for label,count in Counter(df.target.explode()).items():
        if count < 5:
            if 'positive' not in label.split():
                if 'negative' not in label.split():
                    noisy_labels.append(label)

    return noisy_labels

# to remove nosiy labels from the dataframe
def remove_noisy_labels(df):
    noisy_labels = get_noisy_labels(df)
    for i in range(len(df)):
        for nLabel in noisy_labels:
            if nLabel in df.iloc[i,1]:
                df.iloc[i,1].remove(nLabel)

    return df 

# combine labels that have very low frequency to a single label based on threshold
def combine_labels(df,min_samples = 50):
    label_counts = df.target.explode().value_counts()
    label_names = label_counts.index
    
    fewer_labels = []
    for i,label in enumerate(label_names):
        if label_counts[i] < min_samples:
            fewer_labels.append(label)
    
    def replace_fewer(labels):
        fewers = []
        for label in labels:
            sentiment = label.split(' ')[-1]
            if label in fewer_labels:
                fewers.append(' '.join(['extra',sentiment]))
            else:
                fewers.append(label)
                
        return fewers 
    
    df['target'] = df['target'].map(replace_fewer)  

    return df

# encode labels for training
def encode_labels(df):
    le = MultiLabelBinarizer()
    df['encoded'] = le.fit_transform(df.target.tolist()).tolist()
    df = df[['text','encoded']]
    return df 
