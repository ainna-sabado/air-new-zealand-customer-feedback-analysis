import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
from functions import (
    get_sentiment_analysis,
    plot_monthly_sentiment,
    plot_avg_ratings,
    plot_sentiment_distribution,
    plot_avg_ratings_by_seat_type,
    preprocess_text,
    get_top_n_ngrams,
    plot_all_review_ngrams
)
import numpy as np
from PIL import Image

# Load the data
csv_file = '/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/nz_reviews_with_routes.csv'
reviews_df = pd.read_csv(csv_file)

# Preprocess the reviews
reviews_df['cleaned_review'] = reviews_df['review_content'].apply(preprocess_text)

# Parse date
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_df['month'] = reviews_df['date'].dt.month_name()
reviews_df['year'] = reviews_df['date'].dt.year

# Available years for dropdown
available_years = [str(year) for year in sorted(reviews_df['year'].unique())] + ['ALL']
year_selector = pn.widgets.Select(name='Select Year', options=available_years, value='ALL')

# Initial data
initial_year = 'ALL'
year_reviews, monthly_sentiment_counts = get_sentiment_analysis(reviews_df, initial_year)

# Initialize the plots
monthly_sentiment_plot = pn.pane.Matplotlib(plot_monthly_sentiment(monthly_sentiment_counts), sizing_mode='stretch_both')
avg_ratings_plot = pn.pane.Matplotlib(plot_avg_ratings(year_reviews, initial_year), sizing_mode='stretch_both')
sentiment_distribution_plot = pn.pane.Matplotlib(plot_sentiment_distribution(year_reviews, initial_year), sizing_mode='stretch_both')
avg_ratings_seat_plot = pn.pane.Matplotlib(plot_avg_ratings_by_seat_type(year_reviews, initial_year), sizing_mode='stretch_both')

# Get the top n-grams for each sentiment
def get_initial_ngrams():
    positive_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Positive']['cleaned_review']
    negative_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Negative']['cleaned_review']
    neutral_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Neutral']['cleaned_review']
    
    return (
        get_top_n_ngrams(positive_reviews, 20),
        get_top_n_ngrams(negative_reviews, 20),
        get_top_n_ngrams(neutral_reviews, 10)
    )

top_positive_ngrams, top_negative_ngrams, top_neutral_ngrams = get_initial_ngrams()

# Load the airplane mask image
mask = np.array(Image.open('/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/airplane-vector-36294843 copy.jpg'))

# Recommendations (leave blank)
recommendations_pane = pn.pane.Markdown("")
ngram_figures = pn.pane.Matplotlib(plot_all_review_ngrams(top_positive_ngrams, top_negative_ngrams, top_neutral_ngrams, mask), sizing_mode='stretch_width')

# Event listener for year selection
def update_plot(event):
    chosen_year = event.new
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    monthly_sentiment_plot.object = plot_monthly_sentiment(sentiment_counts)
    avg_ratings_plot.object = plot_avg_ratings(year_reviews, chosen_year)
    sentiment_distribution_plot.object = plot_sentiment_distribution(year_reviews, chosen_year)
    avg_ratings_seat_plot.object = plot_avg_ratings_by_seat_type(year_reviews, chosen_year)
    recommendations_pane.object = "No recommendations available."
    ngram_figures.object = plot_all_review_ngrams(year_reviews, chosen_year)

# Watch for changes in the year_selector
year_selector.param.watch(update_plot, 'value')

# Dashboard layout
dashboard_grid = pn.GridSpec(sizing_mode='stretch_both', max_height=2000)

# Create a top row for title and year selector
top_row = pn.Column(
    pn.pane.Markdown("## Air New Zealand Customer Feedback Analysis", sizing_mode='stretch_width'), 
    year_selector
)

# Assign elements to the grid with specified widths
dashboard_grid[0, 1] = monthly_sentiment_plot
dashboard_grid[1, 1] = avg_ratings_plot       
dashboard_grid[2, 1] = sentiment_distribution_plot 
dashboard_grid[3, 1] = avg_ratings_seat_plot  

# Create a separate pane for n-grams plots
#ngram_figures = pn.Column(*plot_all_review_ngrams(top_positive_ngrams, top_negative_ngrams, top_neutral_ngrams, mask))
dashboard_grid[0:4, 2] = ngram_figures  # Place it in the grid

# Recommendations with top row content in the left pane (1st column)
dashboard_grid[:, 0] = pn.Column(
    top_row,  # Include the top row content here
    pn.pane.Markdown("### Recommendations"),
    recommendations_pane
)

# Set sizing mode if needed
dashboard_grid.sizing_mode = 'stretch_both'  # or other sizing modes as required

# Create and serve the dashboard
dashboard = pn.Column(dashboard_grid)
dashboard.show()
