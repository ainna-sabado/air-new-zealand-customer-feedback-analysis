import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud

# Define plot size
PLOT_SIZE = (10, 6)

# Function to parse date and convert it to datetime format
def parse_date(date_str):
    return pd.to_datetime(date_str, format='%Y-%m-%d')

# Function to filter reviews based on selected year
def get_sentiment_analysis(reviews_df, chosen_year):
    if chosen_year == 'ALL':
        year_reviews = reviews_df
        sentiment_counts = year_reviews.groupby(['year', 'vader_sentiment']).size().unstack(fill_value=0)
    else:
        year_reviews = reviews_df[reviews_df['year'] == int(chosen_year)]
        sentiment_counts = year_reviews.groupby(['month', 'vader_sentiment']).size().unstack(fill_value=0)

    if chosen_year != 'ALL':
        sentiment_counts = sentiment_counts.reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])

    return year_reviews, sentiment_counts

# Visualization for monthly sentiment
def plot_monthly_sentiment(counts):
    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}
    
    ax = counts.plot(
        kind='line',
        color=[color_map.get(col, 'black') for col in counts.columns],
        marker='o',
        linewidth=2,
        figsize=PLOT_SIZE
    )
    plt.xlabel('Month' if 'month' in counts.index.name else 'Year', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews', fontsize=14, fontweight='bold')
    plt.title('Total Reviews by Month and Sentiment' if 'month' in counts.index.name else 'Total Reviews by Year and Sentiment', fontsize=16, fontweight='bold')
    plt.legend(title='Sentiment', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

# Function to generate the average ratings plot
def plot_avg_ratings(year_reviews, chosen_year):
    rating_columns = ['seat_comfort', 'cabin_staff_service', 'food_&_beverages', 'ground_service', 
                      'value_for_money', 'inflight_entertainment', 'wifi_&_connectivity']
    average_ratings = year_reviews.groupby(['is_domestic'])[rating_columns].mean().reset_index()
    average_ratings_melted = average_ratings.melt(id_vars=['is_domestic'], value_vars=rating_columns, var_name='category', value_name='average_rating')
    plt.figure(figsize=PLOT_SIZE)
    sns.barplot(data=average_ratings_melted, x='category', y='average_rating', hue='is_domestic')
    plt.title(f'Average Ratings per Category: Domestic vs International Routes ({chosen_year})', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Domestic' if label == 'True' else 'International' for label in labels], title='Route Type', title_fontsize='13', fontsize='12')
    plt.tight_layout()
    return plt.gcf()

def plot_sentiment_distribution(year_reviews, chosen_year):
    sentiment_distribution = year_reviews.groupby(['type_of_traveller', 'vader_sentiment']).size().unstack(fill_value=0)
    sentiment_distribution_percentage = sentiment_distribution.div(sentiment_distribution.sum(axis=1), axis=0)
    
    traveller_types = sentiment_distribution_percentage.index
    n_traveller_types = len(traveller_types)

    # Create a 2x2 subplot grid
    n_rows = 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PLOT_SIZE))
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, traveller_type in enumerate(traveller_types):
        if i < n_rows * n_cols:  # Check to avoid IndexError
            sentiments = sentiment_distribution_percentage.loc[traveller_type]
            axes[i].pie(sentiments, labels=sentiments.index, autopct='%1.1f%%', startangle=90, colors=colors)
            axes[i].set_title(traveller_type)
    
    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis('off')

    # Adjust layout to make room for the suptitle
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust this value as necessary for your layout
    plt.suptitle(f'Sentiment Distribution for Different Types of Travelers ({chosen_year})', fontsize=16, fontweight='bold')
    
    return plt.gcf()

def plot_avg_ratings_by_seat_type(year_reviews, chosen_year):
    # Group by seat type and calculate rating statistics
    seat_rating_summary = year_reviews.groupby('seat_type')['rating'].agg(['mean', 'median', 'count']).reset_index()
    
    # Sort the values to see which seat types have the highest/lowest ratings
    seat_rating_summary = seat_rating_summary.sort_values(by='mean', ascending=False)

    # Create a bar plot to visualize the mean ratings by seat type
    plt.figure(figsize=(PLOT_SIZE))
    sns.barplot(x='seat_type', y='mean', data=seat_rating_summary, palette='coolwarm')

    # Add labels and title
    plt.title(f'Average Ratings by Seat Type ({chosen_year})', fontsize=16, fontweight='bold')
    plt.xlabel('Seat Type')
    plt.ylabel('Average Rating')

    # Display the plot
    plt.tight_layout()
    return plt.gcf()



# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Common words to exclude from analysis
excluded_terms = ["air_new_zealand", "flight", "auckland", "christchurch", "wellington", 
                  "new", "zealand", "air", "nz", "even_though", "via", "av", "sec", "could"]

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Function to generate exactly 3-word n-grams and join words with underscores
def generate_ngrams(text, n=3):
    words = text.split()
    ngrams_list = ["_".join(ngram) for ngram in ngrams(words, n)]
    return ngrams_list

# Function to get top N n-grams while excluding certain terms
def get_top_n_ngrams(sentiment_reviews, n=20):
    all_ngrams = []
    for review in sentiment_reviews:
        all_ngrams.extend(generate_ngrams(review)) 
    
    filtered_ngrams = [ngram for ngram in all_ngrams if all(term not in ngram for term in excluded_terms)]
    ngram_freq = Counter(filtered_ngrams)
    return ngram_freq.most_common(n)

# Function to preprocess n-grams
def preprocess_ngrams(ngram_freq):
    word_list = []
    for ngram, freq in ngram_freq.items():
        words = ngram.split('_')
        word_list.extend(words * freq)
    return Counter(word_list)

# Function to plot the word cloud
def plot_wordcloud(ngram_freq, title, mask):
    if not ngram_freq:  
        print(f"No n-grams to plot for {title}.")
        return
    
    word_freq_dict = preprocess_ngrams(ngram_freq)
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis', mask=mask).generate_from_frequencies(word_freq_dict)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.show()
