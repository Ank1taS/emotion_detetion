import re
import pickle


# data cleaning
def get_clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]*', '', text)               # to remove @mentions
    text = re.sub(r'#', '', text)                           # to remove # tag
    text = re.sub(r'RT[\s]+', '', text)                     # to remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)              # to remove hyperlinks
    text = re.sub('(\\\\u([a-z]|[0-9])+)', '', text)        # to remove unicode escape sequence character
    # text = re.sub(r'\\\\u([0-9A-Fa-f]{4})', '', text)       # to remove unicode escape sequence character
    text = re.sub(r'"', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'https?:?', '', text)
    text = re.sub(r'href', '', text)

    return text

text_clean_function = get_clean_text
pickle.dump(text_clean_function, open('clean_text.pkl', 'wb'))

