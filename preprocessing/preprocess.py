from datasets import load_dataset
import pandas as pd

def load_data_from_dataset(dataset_name): 
    # Load data from dataset and returns pandas dataframe
    dataset = load_dataset(dataset_name)

    return dataset

#- filter text with emojis, special characters, emails, urls, aphosotrophes, but keep stopwords for the synthetisizer to sound more natural
def remove_special_tokens(data):

    # Remove emojis
    data['text'] = data['text'].replace('[^\x00-\x7F]+', '', regex=True)
    # Remove emails, hashtags, mentions
    data['text'] = data['text'].replace('\S*@\S*\s?', '', regex=True)
    data['text'] = data['text'].replace('\S*#\S*\s?', '', regex=True)

    # Remove URL
    data['text'] = data['text'].replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)
    data['text'] = data['text'].replace('www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)
    data['text'] = data['text'].replace('URL', '', regex=True)
    data['text'] = data['text'].replace('url', '', regex=True)

    #remove , at the first
    data['text'] = data['text'].replace(',,', '', regex=True)
    data['text'] = data['text'].replace(',(?=\s*[A-Z])', '', regex=True)

    data['text'] = data['text'].replace('"', '', regex=True)
    # Remove special characters but keep punctuation etc..
    data['text'] = data['text'].replace('[^a-zA-Z0-9\s.,!?;:\']', '', regex=True)
 
    return data

def remove_non_english(data):
    data['text'] = data['text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    return data

def replace_abbreviations_and_slang(data):
    data_abreviations = pd.read_csv('preprocessing/abbreviations_and_slang.csv')

    data['text'] = data['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)

    for _, row in data_abreviations.iterrows():
        data['text'] = data['text'].replace("\\b"+row['Abbreviations']+"\\b", str(row['Text']).lower(), regex=True)

    return data 