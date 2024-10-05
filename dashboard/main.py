import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
import re
from PIL import Image
from wordcloud import WordCloud
import holoviews as hv
from bokeh.models import HoverTool

# Ensure Panel is using the latest template
pn.extension()

hv.extension('bokeh')

# Set the CSV file path and year for analysis
csv_file = '/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/nz_reviews_with_routes.csv'

# Load the dataset
reviews_df = pd.read_csv(csv_file)

# Convert date column to datetime and extract the year
reviews_df['date'] = pd.to_datetime(reviews_df['date'], format='%Y-%m-%d')
reviews_df['year'] = reviews_df['date'].dt.year
reviews_df['month'] = reviews_df['date'].dt.strftime('%B')

# Available years for dropdown
available_years = [str(year) for year in sorted(reviews_df['year'].unique())] + ['ALL']

# Overall average rating
average_rating = reviews_df['rating'].mean()

rating_html = f"""
<div style="text-align: center; font-size: 24px; color: #333; background-color: #f8f9fa; border-radius: 8px; padding: 10px;">
    <strong>Overall Rating:</strong> {average_rating:.2f} / 10
</div>
"""

rating_display = pn.pane.HTML(rating_html, width=300, height=100)

# Rating categories
rating_columns = ['seat_comfort', 'cabin_staff_service', 'food_&_beverages', 'ground_service',
                  'wifi_&_connectivity', 'value_for_money', 'inflight_entertainment']

# Calculate average ratings by category
average_ratings_by_category = np.ceil(reviews_df[rating_columns].mean())

top_5_ratings = average_ratings_by_category.sort_values(ascending=False).head(5)
top_5_ratings = top_5_ratings.sort_index()

# Function to generate stars for ratings
def generate_star_html(rating):
    full_star = '<span style="color: gold; font-size: 24px;">&#9733;</span>'
    half_star = '<span style="color: gold; font-size: 24px;">&#11088;</span>'
    empty_star = '<span style="color: lightgray; font-size: 24px;">&#9734;</span>'
    
    stars = ""
    full_stars = int(rating)
    half_stars = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_stars

    stars += full_star * full_stars
    stars += half_star * half_stars
    stars += empty_star * empty_stars
    
    return stars

# Generate the HTML for all categories
html_content = "<div style='font-family: Arial, sans-serif; padding: 10px;'>"
html_content += "<h3>Overall Average Ratings by Category</h3>"

for category, rating in top_5_ratings.items():
    stars = generate_star_html(rating)
    html_content += f"<div style='margin-bottom: 10px;'><strong>{category.replace('_', ' ').title()}:</strong> {stars} ({rating:.1f})</div>"

html_content += "</div>"

overall_category_ratings = pn.pane.HTML(html_content, width=500)

year_selector = pn.widgets.Select(name='Select Year', options=available_years, value='ALL', sizing_mode='stretch_width')

# Ratings converted to numeric if they are in string format
reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')

def get_sentiment_analysis(reviews_df, chosen_year):
    if chosen_year == 'ALL':
        year_reviews = reviews_df
        sentiment_counts = year_reviews.groupby(['year', 'vader_sentiment']).size().unstack(fill_value=0)
    else:
        year_reviews = reviews_df[reviews_df['year'] == int(chosen_year)]
        sentiment_counts = year_reviews.groupby(['month', 'vader_sentiment']).size().unstack(fill_value=0)
        sentiment_counts = sentiment_counts.reindex(
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December'], fill_value=0)

    return year_reviews, sentiment_counts

# Visualization for monthly or yearly sentiment
def plot_monthly_sentiment_hv(sentiment_counts, chosen_year):
    sentiment_counts = sentiment_counts.reset_index()

    sentiment_counts_melted = sentiment_counts.melt(
        id_vars=['month' if 'month' in sentiment_counts.columns else 'year'], 
        var_name='Sentiment', 
        value_name='Count'
    )

    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}

    sentiment_plots = hv.NdOverlay({
        sentiment: hv.Curve(
            sentiment_counts_melted[sentiment_counts_melted['Sentiment'] == sentiment], 
            kdims=['month' if 'month' in sentiment_counts_melted.columns else 'year'], 
            vdims=['Count'],
            label=sentiment
        ).opts(color=color_map[sentiment], line_width=2)
        for sentiment in sentiment_counts_melted['Sentiment'].unique()
    })

    sentiment_plots = sentiment_plots.opts(
        xlabel='Month' if 'month' in sentiment_counts.index.names else 'Year',
        ylabel='Number of Reviews',
        tools=[HoverTool(tooltips=[('Month/Year', '@x'), ('Count', '@y')])],
        show_legend=True,
        legend_position='top_left',
        width=1000,
        height=600
    )
    
    return sentiment_plots


def plot_avg_ratings(year_reviews, chosen_year):
    rating_columns = [
        'seat_comfort', 'cabin_staff_service', 'food_&_beverages', 
        'ground_service', 'value_for_money', 'inflight_entertainment', 
        'wifi_&_connectivity'
    ]
    
    # Calculate average ratings
    average_ratings = year_reviews.groupby('is_domestic')[rating_columns].mean().reset_index()
    
    # Melt the DataFrame for easier plotting
    average_ratings_melted = average_ratings.melt(
        id_vars='is_domestic', 
        value_vars=rating_columns, 
        var_name='category', 
        value_name='average_rating'
    )
    
    # Map 'is_domestic' to readable route types
    average_ratings_melted['Route Type'] = average_ratings_melted['is_domestic'].map({True: 'Domestic', False: 'International'})
    
    # Sort the data so that bars with higher average ratings are plotted first
    average_ratings_melted = average_ratings_melted.sort_values(by=['category', 'average_rating'], ascending=[True, False])
    
    # Define color mapping
    color_mapping = {'Domestic': '#1f77b4', 'International': '#ff7f0e'}
    
    # Ensure categories follow the original order
    average_ratings_melted['category'] = pd.Categorical(
        average_ratings_melted['category'], 
        categories=rating_columns, 
        ordered=True
    )
    
    # Create the bar plot using Holoviews
    bars = hv.Bars(
        average_ratings_melted, 
        kdims=['category'], 
        vdims=['average_rating', 'Route Type']
    )
    
    # Set options for the bar plot
    bars.opts(
        xlabel='Category',
        ylabel='Average Rating',
        color='Route Type',
        cmap=list(color_mapping.values()),
        legend_position='top_right',
        show_legend=True,
        width=800,
        height=500,
        tools=['hover'],
        hover_tooltips=[
            ('Category', '@category'), 
            ('Route Type', '@{Route Type}'), 
            ('Average Rating', '@{average_rating}{0.2f}')  # Format to 2 decimal places
        ],
        bar_width=0.9,
        line_color='black',
        line_width=1,
        title=f'Average Ratings per Category: Domestic vs International ({chosen_year})'
    )
    
    return bars

# Create buttons for the sidebar navigation
button_home = pn.widgets.Button(name="Home", button_type="primary")
button1 = pn.widgets.Button(name="Overall Sentiment Analysis", button_type="primary")
button2 = pn.widgets.Button(name="Average Ratings per Category", button_type="primary")

main_area = pn.Column()

def show_page1(event=None):
    chosen_year = year_selector.value  
    main_area.clear() 
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    sentiment_plots = plot_monthly_sentiment_hv(sentiment_counts, chosen_year)
    main_area.append(pn.pane.HoloViews(sentiment_plots))

def show_page2(event=None):
    chosen_year = year_selector.value 
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.clear()
    avg_ratings_plot = plot_avg_ratings(year_reviews, chosen_year)
    main_area.append(pn.pane.HoloViews(avg_ratings_plot, height=600))

def show_home_page(event=None):
    chosen_year = year_selector.value  
    main_area.clear()  
    grid = pn.GridSpec(sizing_mode='stretch_both')
    grid[0, 2] = pn.Column(rating_display, overall_category_ratings)  
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    sentiment_plots = plot_monthly_sentiment_hv(sentiment_counts, chosen_year)
    grid[0, 0:1] = pn.pane.HoloViews(sentiment_plots, height=500, width=900)  
    grid[2, 0] = pn.pane.Matplotlib(plot_avg_ratings(year_reviews, chosen_year), height=400)  

button_home.on_click(lambda event: set_current_page('home'))
button1.on_click(lambda event: set_current_page('page1'))
button2.on_click(lambda event: set_current_page('page2'))

def set_current_page(page):
    global current_page
    current_page = page
    if page == 'home':
        show_home_page()
    elif page == 'page1':
        show_page1()
    elif page == 'page2':
        show_page2()

def update_on_year_change(event):
    if current_page == 'home':
        show_home_page()
    elif current_page == 'page1':
        show_page1()
    elif current_page == 'page2':
        show_page2()

year_selector.param.watch(update_on_year_change, 'value')

sidebar = pn.Column(
    pn.pane.Markdown("## Pages"),
    year_selector,
    button_home,  
    button1,
    button2,
    styles={"width": "100%", "padding": "15px"}
)

dashboard = pn.template.BootstrapTemplate(
    title="Customer Feedback Analysis",
    sidebar=[sidebar],
    main=[main_area],
    header_background="black", 
    site="Air New Zealand",
    theme=pn.template.DarkTheme,
    sidebar_width=250,
)

dashboard.show()

set_current_page('home')
