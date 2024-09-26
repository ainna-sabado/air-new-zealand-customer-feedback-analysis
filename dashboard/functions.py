import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
import re
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

# Function to generate n-grams
def generate_ngrams(text, n=3):
    words = text.split()
    ngrams_list = ["_".join(ngram) for ngram in ngrams(words, n)]
    return ngrams_list

# Function to get top N n-grams while excluding certain terms
def get_top_n_ngrams(sentiment_reviews, n=20):
    all_ngrams = []
    for review in sentiment_reviews:
        all_ngrams.extend(generate_ngrams(review)) 
    
    # Remove excluded terms
    filtered_ngrams = [ngram for ngram in all_ngrams if all(term not in ngram for term in excluded_terms)]
    
    # Count frequencies of remaining n-grams
    ngram_freq = Counter(filtered_ngrams)
    
    # Return a dictionary instead of a list of tuples
    return dict(ngram_freq.most_common(n))  # Return as a dictionary

def preprocess_ngrams(ngram_freq):
    # Split multi-word n-grams into individual words
    word_list = []
    for ngram, freq in ngram_freq.items():  # This will now work because ngram_freq is a dictionary
        words = ngram.split('_')
        word_list.extend(words * freq)
    return Counter(word_list)

def plot_wordcloud(ngram_freq, title, mask, ax):
    if not ngram_freq:  
        ax.set_title(f'Word Cloud for {title}')
        ax.axis('off')  # Hide axis if no n-grams
        return
    
    # Preprocess n-grams to individual words
    word_freq_dict = preprocess_ngrams(ngram_freq)
    
    # Create WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', mask=mask).generate_from_frequencies(word_freq_dict)
    
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # Hide axes
    ax.set_title(f'Word Cloud for {title}')

def plot_ngrams(ngram_freq, title, ax):
    if not ngram_freq:
        ax.set_title(f'Top N-grams for {title}')
        ax.axis('off')  # Hide axis if no n-grams
        return
    
    ngrams = list(ngram_freq.keys())  # Get n-grams as a list
    counts = list(ngram_freq.values())  # Get counts as a list

    ax.barh(ngrams, counts, color='skyblue')
    ax.set_title(f'Top N-grams for {title}')
    ax.invert_yaxis()  # Invert y-axis to have the highest count on top
    ax.set_xlabel('Frequency')


def plot_all_review_ngrams(top_positive_ngrams, top_negative_ngrams, top_neutral_ngrams, mask):
    """Plots word clouds and n-grams for positive, negative, and neutral reviews."""
    
    # Create a single figure with subplots for all sentiments
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))  # 3 rows for each sentiment, 2 columns for wordcloud and n-grams
    
    # Function to safely plot with handling for empty n-grams
    def safe_plot(wordcloud_ngrams, sentiment_title, ax_wordcloud, ax_ngrams):
        if wordcloud_ngrams:
            plot_wordcloud(wordcloud_ngrams, sentiment_title, mask, ax_wordcloud)
            plot_ngrams(wordcloud_ngrams, f'Top N-grams - {sentiment_title}', ax_ngrams)
        else:
            ax_wordcloud.set_title(f'No N-grams for {sentiment_title}')
            ax_wordcloud.axis('off')  # Hide the axis for the word cloud
            ax_ngrams.set_title(f'No N-grams for {sentiment_title}')
            ax_ngrams.axis('off')  # Hide the axis for the n-grams

    # Plotting for Positive Reviews
    safe_plot(top_positive_ngrams, 'Positive Reviews', axes[0, 0], axes[0, 1])

    # Plotting for Negative Reviews
    safe_plot(top_negative_ngrams, 'Negative Reviews', axes[1, 0], axes[1, 1])

    # Plotting for Neutral Reviews
    safe_plot(top_neutral_ngrams, 'Neutral Reviews', axes[2, 0], axes[2, 1])

    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()  # Return the single combined figure

