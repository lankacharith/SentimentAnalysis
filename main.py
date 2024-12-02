import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pprint

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

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