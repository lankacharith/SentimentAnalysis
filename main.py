import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import transformers

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import pipeline

plt.style.use('ggplot')

# Reading the data
df = pd.read_csv('/Users/charithlanka/SentimentAnalysis/archive/Reviews.csv')

# Top 500 reviews
df = df.head(500)

# Data Analysis
ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

# NLTK
example = df['Text'][50]

tokens = nltk.word_tokenize(example)
tokens[:10]

tagged = nltk.pos_tag(tokens)
tagged[:10]

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# VADER Sentiment Scoring
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(example))

# Run the polarity score on the entire dataset
res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

# Plot VADER results
ax = sns.barplot(data=vaders, x='Score', y='compound') 
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
axs[0].set_title('Positive')

sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
axs[1].set_title('Neutral')

sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[2].set_title('Negative')

plt.tight_layout()
plt.show()

# Roberta Pretrained Model
transformers.logging.set_verbosity_error() # Suppress transformer warnings

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    scores_dict = {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2])
    }
    
    return scores_dict

res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text'] 
        myid = row['Id']

        vader_result = sia.polarity_scores(text) 

        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f'vader_{key}'] = value

        roberta_result = polarity_scores_roberta(text)

        both = {**vader_result_rename, **roberta_result}
        res[myid] = both

    except RuntimeError:
        print(f'Broke for id {myid}')

print(both)

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Compare scores between models
sns.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'], hue='Score', palette='tab10')
plt.show()

# Review examples
print(results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)[['Text']].values[0])
print(results_df.query('Score == 1').sort_values('vader_pos', ascending=False)[[ 'Text']].values[0])

# # Negative sentiment 5 star review
print(results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)[['Text']].values[0])
print(results_df.query('Score == 5').sort_values('vader_neg', ascending=False)[[ 'Text']].values[0])

# Transformers pipeline
sentiment_pipeline = pipeline('sentiment-analysis')
print(sentiment_pipeline('I love sentiment analysis!'))
print(sentiment_pipeline('I am very angry about this!'))
print(sentiment_pipeline('This is a neutral statement.'))