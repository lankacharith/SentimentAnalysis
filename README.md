# Sentiment Analysis Project

This project explores sentiment analysis using a combination of natural language processing (NLP) techniques and machine learning models. It leverages the **VADER Sentiment Analysis** tool and a **pre-trained RoBERTa model** to analyze Amazon review data and compare sentiment scoring methodologies.

## Overview

The primary objectives of this project are:
- Analyzing sentiment in customer reviews using VADER and RoBERTa models.
- Comparing sentiment scores generated by each model.
- Visualizing sentiment trends across different star ratings.

The dataset used consists of the first 500 reviews from an Amazon product reviews dataset, providing a manageable yet insightful scope for analysis.


## Key Features

### 1. **Data Exploration**
- **Visualization:** Plots the distribution of review ratings (star counts) to understand the dataset composition.
- **Text Tokenization:** Processes text data for sentiment analysis using NLTK.

### 2. **Sentiment Analysis with VADER**
- Sentiment scoring is performed using the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool.
- Scores are categorized into **Positive**, **Neutral**, **Negative**, and a **Compound** summary.

### 3. **Sentiment Analysis with RoBERTa**
- Utilizes the Hugging Face **cardiffnlp/twitter-roberta-base-sentiment-latest** model.
- Generates probabilities for negative, neutral, and positive sentiment.
- Scores are derived using a softmax function.

### 4. **Comparison of Models**
- Merges results from VADER and RoBERTa for direct comparison.
- Pairplots visualize relationships between sentiment scores across models.

### 5. **Visualization**
- Bar plots for sentiment trends across star ratings.
- Pairplots for multi-model sentiment score correlations.


## Results

- Sentiment trends are examined across 1- to 5-star ratings.
- Significant differences between VADER and RoBERTa sentiment scores are highlighted through visualizations.
- Review-specific examples are used to showcase discrepancies in sentiment interpretation by the models.


## Dependencies

This project requires the following libraries:

- **Python:** Data manipulation and analysis
- **Pandas:** Dataframe operations
- **NumPy:** Numerical computing
- **Matplotlib & Seaborn:** Visualization
- **NLTK:** Natural language processing (tokenization, POS tagging, chunking)
- **Transformers (Hugging Face):** Model loading and sentiment analysis
- **Tqdm:** Progress bars


## Installation

To run the code, install the dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn nltk transformers tqdm