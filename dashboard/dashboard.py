# Data manipulation and analysis
import pandas as pd
import numpy as np
from collections import Counter
import re

# Visualization
import panel as pn
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv
from wordcloud import WordCloud
from bokeh.models import HoverTool, Select, ColumnDataSource, Div
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column
from bokeh.palettes import Category20
from bokeh.transform import cumsum

# Natural Language Processing
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))


# Additional imports for threading and timing
import time
import threading
from PIL import Image

import subprocess
import io


pn.extension()
hv.extension('bokeh')

############################ LOAD DATASETS ############################
# Set the CSV file path and year for analysis
csv_file = 'nz_reviews_with_routes.csv'

# Load the dataset
reviews_df = pd.read_csv(csv_file)

# Convert date column to datetime and extract the year
reviews_df['date'] = pd.to_datetime(reviews_df['date'], format='%Y-%m-%d')
reviews_df['year'] = reviews_df['date'].dt.year
reviews_df['month'] = reviews_df['date'].dt.strftime('%B')

# Available years for dropdown
available_years = [str(year) for year in sorted(reviews_df['year'].unique())] + ['ALL']


############################ OVERALL RESULTS ############################
#all_reviews = "air_nz_cleaned_data.csv"
#all_reviews = pd.read_csv(all_reviews)

# Overall average rating
average_rating = reviews_df['rating'].mean()

rating_html = """
<div style="text-align: center; font-size: 24px; color: #333; background-color: #f8f9fa; border-radius: 8px; padding: 10px;">
    <strong>Overall Rating:</strong> {average_rating:.2f} / 10
</div>
""".format(average_rating=average_rating)

rating_display = pn.pane.HTML(rating_html, width=300)

# Rating categories
rating_columns = ['seat_comfort', 'cabin_staff_service', 'food_&_beverages', 'ground_service', 
                  'wifi_&_connectivity', 'value_for_money', 'inflight_entertainment']

average_ratings_by_category = np.ceil(reviews_df[rating_columns].mean())
top_5_ratings = average_ratings_by_category.sort_values(ascending=False).head(5)
top_5_ratings = top_5_ratings.sort_index()

# Define star symbols
def generate_star_html(rating):
    full_star = '<span style="color: gold; font-size: 24px;">&#9733;</span>'  # Filled star (★)
    half_star = '<span style="color: gold; font-size: 24px;">&#11088;</span>'  # Half star
    empty_star = '<span style="color: lightgray; font-size: 24px;">&#9734;</span>'  # Empty star (☆)

    stars = ""
    full_stars = int(rating)  # Full stars count
    half_stars = 1 if (rating - full_stars) >= 0.5 else 0  # Half star if the remainder is >= 0.5
    empty_stars = 5 - full_stars - half_stars  # Remaining stars are empty

    stars += full_star * full_stars  # Add full stars
    stars += half_star * half_stars  # Add half star
    stars += empty_star * empty_stars  # Add empty stars
    
    return stars

# Generate HTML for category ratings in a table
html_content = """
<div style='font-family: Arial, sans-serif; padding: 10px;'>
    <table style='width: 100%; border-collapse: collapse; background-color: white;'>
        <thead>
            <tr>
                <th colspan="2" style='text-align: center; padding: 8px; border: 1px solid #ddd; color: black; font-size: 18px;'>
                    Overall Average Ratings by Category
                </th>
            </tr>
        </thead>
        <tbody>
"""

for category, rating in top_5_ratings.items():
    stars = generate_star_html(rating)
    html_content += f"""
            <tr>
                <td style='padding: 3px; border: 1px solid #ddd; color: black;'><strong>{category.replace('_', ' ').title()}</strong></td>
                <td style='padding: 3px; border: 1px solid #ddd; color: black;'>{stars}</td>
            </tr>
    """

html_content += """
        </tbody>
    </table>
</div>
"""

overall_category_ratings = pn.pane.HTML(html_content, width=300)

# Calculate the percentage of recommendations
total_recommendations = reviews_df['recommended'].count()
recommended_count = reviews_df['recommended'].sum()  # True values are counted as 1
not_recommended_count = total_recommendations - recommended_count

# Calculate percentages
if total_recommendations > 0:
    thumbs_up_percentage = (recommended_count / total_recommendations) * 100
    thumbs_down_percentage = (not_recommended_count / total_recommendations) * 100
else:
    thumbs_up_percentage = 0
    thumbs_down_percentage = 0

# Create HTML for thumbs up and thumbs down
thumbs_html = f"""
<div style="text-align: center; font-size: 20px; color: #333; background-color: #f8f9fa; border-radius: 8px; padding: 10px;">
    <div>
        <span style="font-size: 20px; color: green;">&#128077;</span> 
        <strong>{thumbs_up_percentage:.1f}%</strong>
        &nbsp; &nbsp;  <!-- Adds some space between thumbs -->
        <span style="font-size: 20px; color: red;">&#128078;</span> 
        <strong>{thumbs_down_percentage:.1f}%</strong>
    </div>
</div>
"""

thumbs_display = pn.pane.HTML(thumbs_html, width=300)

# Create a combined container for all displays
combined_html = f"""
<div style='padding: 5px; background-color: #f8f9fa; border-radius: 8px; width: 300px;'>
    {rating_display.object}
    {overall_category_ratings.object}
    {thumbs_display.object}
</div>
"""

# Create the final pane for display
final_display = pn.pane.HTML(combined_html, width=300)


############################ WIDGET FOR YEAR SELECTION ############################
year_selector = pn.widgets.Select(name='Select Year', options=available_years, value='ALL', sizing_mode='stretch_width')

############################ REFRESH DASHBOARD ############################

# Alert message
text = "Are you sure you want to refresh? This process will take 10 to 20 minutes to load. Please note that it involves downloading data from SkyTrax (airlinequality.com)."

# Create a Button in the dashboard
refresh_button = pn.widgets.Button(name='Refresh Data', button_type='warning', button_style='outline', icon='refresh', sizing_mode='stretch_width')

# Create alert pane and buttons for confirmation
alert_pane = pn.pane.Alert(text, alert_type='warning')
confirm_button = pn.widgets.Button(name='Yes', button_type='success')
cancel_button = pn.widgets.Button(name='No', button_type='danger')

# Create a row to hold the buttons
buttons_row = pn.Row(confirm_button, cancel_button, align='center')

# Create a column to hold the alert and buttons
alert_panel = pn.Column(alert_pane, buttons_row)

# Set the alert panel to be hidden initially
alert_panel.visible = False

# Function to refresh the dashboard
def refresh_dashboard(event):
    try:
        #Step 1: Run the web-scraping notebook
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", 
             "web-scraping.ipynb"],
            check=True
        )

        # Step 2: Run the EDA notebook
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", 
             "eda-customer-feedback.ipynb"],
            check=True
        )

        # Step 3: Temporarily change the button label to indicate refresh
        refresh_button.name = "Refreshing..."
        
        # Step 4: Reload the dataset
        global reviews_df
        reviews_df = pd.read_csv(csv_file)  # Reload the data after processing

        # Convert date column to datetime and extract the year
        reviews_df['date'] = pd.to_datetime(reviews_df['date'], format='%Y-%m-%d')
        reviews_df['year'] = reviews_df['date'].dt.year
        reviews_df['month'] = reviews_df['date'].dt.strftime('%B')

        # Available years for dropdown
        available_years = [str(year) for year in sorted(reviews_df['year'].unique())] + ['ALL']
 
        time.sleep(1)  # Simulate a refresh action
        refresh_button.name = "Refresh Data"  
        
        show_home_page()  
        
    except subprocess.CalledProcessError as e:
        print(f"Error running notebook: {e}")

# Function to handle the refresh button click
def show_alert(event):
    alert_panel.visible = True  # Show the alert panel

# Attach the show_alert function to the refresh_button's click event
refresh_button.on_click(show_alert)

# Callback for confirmation button
def confirm_refresh(event):
    refresh_dashboard(event)  # Refresh the dashboard
    alert_panel.visible = False  # Hide the alert panel after confirmation

# Callback for cancel button
def cancel_refresh(event):
    alert_panel.visible = False  # Hide the alert panel without doing anything

# Attach callbacks to the buttons
confirm_button.on_click(confirm_refresh)
cancel_button.on_click(cancel_refresh)

############################ CREATE CHARTS ############################
current_page = 'home'

reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')

def get_sentiment_analysis(reviews_df, chosen_year):
    if chosen_year == 'ALL':
        year_reviews = reviews_df
        sentiment_counts = year_reviews.groupby(['year', 'vader_sentiment']).size().unstack(fill_value=0)
    else:
        year_reviews = reviews_df[reviews_df['year'] == int(chosen_year)]
        sentiment_counts = year_reviews.groupby(['month', 'vader_sentiment']).size().unstack(fill_value=0)

        sentiment_counts = sentiment_counts.reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ], fill_value=0)

    total_reviews = year_reviews.shape[0]
    return year_reviews, sentiment_counts, total_reviews

### TOTAL NUMBER OF REVIEWS BY MONTH/YEAR AND SENTIMENT
def plot_monthly_sentiment(sentiment_counts, chosen_year, total_reviews):
    sentiment_counts = sentiment_counts.reset_index()
    time_column = 'month' if 'month' in sentiment_counts.columns else 'year'
    sentiment_counts_melted = sentiment_counts.melt(
        id_vars=[time_column],
        var_name='Sentiment', 
        value_name='Count'
    )

    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}
    sentiment_plots = hv.NdOverlay({
        sentiment: hv.Curve(
            sentiment_counts_melted[sentiment_counts_melted['Sentiment'] == sentiment], 
            kdims=[time_column],  # Use time_column for kdims
            vdims=['Count'],
            label=sentiment
        ).opts(
            color=color_map[sentiment], line_width=2
        )
        for sentiment in sentiment_counts_melted['Sentiment'].unique()
    })

    xlabel = 'Month' if time_column == 'month' else 'Year'

    sentiment_plots = sentiment_plots.opts(
        xlabel=xlabel,  
        ylabel='Number of Reviews',
        tools=[HoverTool(tooltips=[('Month/Year', '@x'), ('Count', '@y')])],
        show_legend=True,
        legend_position='top_left',
        title=f'Total Number of Reviews ({total_reviews}) by {xlabel} and Sentiment ({chosen_year})',  
        height=600,
        width=900,  
        show_grid=True
    )

    return sentiment_plots

### AVERAGE RATINGS PER CATEGORY: DOMESTIC VS INTERNATIONAL ROUTES
def plot_avg_ratings(year_reviews, chosen_year):
    rating_columns = [
        'seat_comfort', 'cabin_staff_service', 'food_&_beverages', 
        'ground_service', 'value_for_money', 'inflight_entertainment', 
        'wifi_&_connectivity'
    ]
    
    average_ratings = year_reviews.groupby('is_domestic')[rating_columns].mean().reset_index()
    average_ratings_melted = average_ratings.melt(
        id_vars='is_domestic', 
        value_vars=rating_columns, 
        var_name='category', 
        value_name='average_rating'
    )
    
    average_ratings_melted['Route Type'] = average_ratings_melted['is_domestic'].map({True: 'Domestic', False: 'International'})
    
    average_ratings_melted['category'] = (
        average_ratings_melted['category']
        .str.replace('_', ' ')  
        .str.title()  
    )
    
    # Sort by average_rating only, highest to lowest
    average_ratings_melted = average_ratings_melted.sort_values(by='average_rating', ascending=False)

    color_mapping = {'Domestic': '#1f77b4', 'International': '#ff7f0e'}
    
    average_ratings_melted['category'] = pd.Categorical(
        average_ratings_melted['category'], 
        categories=[col.replace('_', ' ').title() for col in rating_columns], 
        ordered=True
    )
    
    bars = hv.Bars(
        average_ratings_melted, 
        kdims=['category'], 
        vdims=['average_rating', 'Route Type']
    )
    
    bars.opts(
        ylabel='Average Rating', 
        xlabel='Category',  
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
            ('Average Rating', '@{average_rating}{0.2f}')  
        ],
        bar_width=0.9,
        line_color='black',
        line_width=1,
        title=f'Average Ratings per Category:\nDomestic vs International ({chosen_year})',  
        invert_axes=True 
    )

    return bars



### SENTIMENT DISTRIBUTION BY TRAVELER TYPE
def plot_traveller_sentiments(year_reviews, chosen_year):
    """
    Generate a pie chart for sentiment distribution by type of traveler using Bokeh.
    """

    sentiment_distribution = year_reviews.groupby(['type_of_traveller', 'vader_sentiment']).size().unstack(fill_value=0)
    traveller_types = sentiment_distribution.index.tolist()
    current_type = traveller_types[0]
    sentiments = sentiment_distribution.loc[current_type]
    data = pd.DataFrame(sentiments).reset_index()
    data.columns = ['vader_sentiment', 'value']
    data['angle'] = data['value'] / data['value'].sum() * 2 * np.pi

    sentiment_color_mapping = {
        'Negative': '#FF5733', # (red)
        'Positive': '#28A745',  # (green)
        'Neutral': '#FFC107'     # (yellow)
    }
    
    data['color'] = data['vader_sentiment'].map(sentiment_color_mapping)
    data['percentage'] = ((data['value'] / data['value'].sum()) * 100).round(2).astype(str) 

    source = ColumnDataSource(data)

    pie_chart = figure(height=325, width=600, title=current_type, toolbar_location=None,
                       tools="hover", tooltips="@vader_sentiment: @percentage%", x_range=(-0.5, 1.0))

    pie_chart.wedge(x=0, y=1, radius=0.3, 
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    line_color="white", fill_color='color', legend_field='vader_sentiment', source=source)

    pie_chart.axis.visible = False
    pie_chart.grid.grid_line_color = None

    dropdown = Select(title="Type of Traveler", value=current_type, options=traveller_types)

    def update(attr, old, new):
        current_type = dropdown.value
        sentiments = sentiment_distribution.loc[current_type]
        data = pd.DataFrame(sentiments).reset_index()
        data.columns = ['vader_sentiment', 'value']
        data['angle'] = data['value'] / data['value'].sum() * 2 * np.pi
        data['color'] = data['vader_sentiment'].map(sentiment_color_mapping)
        data['percentage'] = ((data['value'] / data['value'].sum()) * 100).round(2).astype(str) 

        source.data = {
            'vader_sentiment': data['vader_sentiment'],
            'value': data['value'],
            'angle': data['angle'],
            'color': data['color'],
            'percentage': data['percentage']
        }

        pie_chart.title.text = current_type

    dropdown.on_change('value', update)
    layout = column(dropdown, pie_chart)

    return layout

### AVERAGE RATINGS PER SEAT TYPE
def plot_seat_type_ratings(year_reviews, chosen_year):
    seat_rating_summary = year_reviews.groupby('seat_type')['rating'].agg(['mean', 'median', 'count']).reset_index()
    seat_rating_summary = seat_rating_summary.sort_values(by='mean', ascending=False)

    bar_plot = hv.Bars(seat_rating_summary, kdims=['seat_type'], vdims=['mean']).opts(
        title=f'Average Ratings by Seat Type ({chosen_year})',
        xlabel='Seat Type',
        ylabel='Average Rating',
        color='seat_type',
        cmap='coolwarm',
        tools=['hover'],
        width=600,
        height=400
    )

    return bar_plot

### N-GRAM AND WORDCLOUD SENTIMENT ANALYSIS
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

excluded_terms = ["air_new_zealand", "flight", "auckland", "christchurch", "wellington", 
                  "new", "zealand", "air", "nz", "even_though", "via", "av", "sec", "could"]

mask = np.array(Image.open('airplane-vector-36294843 copy.jpg'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def generate_ngrams(text, n=3):
    words = text.split()
    ngrams_list = ["_".join(ngram) for ngram in ngrams(words, n)]
    return ngrams_list

def get_top_n_ngrams(sentiment_reviews, n=20):
    all_ngrams = []
    for review in sentiment_reviews:
        all_ngrams.extend(generate_ngrams(review)) 
    
    filtered_ngrams = [ngram for ngram in all_ngrams if all(term not in ngram for term in excluded_terms)]
    ngram_freq = Counter(filtered_ngrams)

    return ngram_freq.most_common(n)

reviews_df['cleaned_review'] = reviews_df['review_content'].apply(preprocess_text)

positive_reviews = reviews_df[reviews_df['vader_sentiment'] == 'Positive']['cleaned_review']
negative_reviews = reviews_df[reviews_df['vader_sentiment'] == 'Negative']['cleaned_review']
neutral_reviews = reviews_df[reviews_df['vader_sentiment'] == 'Neutral']['cleaned_review']

top_positive_ngrams = get_top_n_ngrams(positive_reviews, 20)
top_negative_ngrams = get_top_n_ngrams(negative_reviews, 20)
top_neutral_ngrams = get_top_n_ngrams(neutral_reviews, 10)

def preprocess_ngrams(ngram_freq):
    word_list = []
    for ngram, freq in ngram_freq:
        words = ngram.split('_')
        word_list.extend(words * freq)
    return Counter(word_list)

def plot_wordcloud(ngram_freq, title, mask, ax):
    if not ngram_freq:  
        ax.set_title(f'Word Cloud for {title}')
        ax.axis('off')
        return
    
    word_freq_dict = preprocess_ngrams(ngram_freq)
    wordcloud = WordCloud(width=800, height=400, background_color='white', mask=mask).generate_from_frequencies(word_freq_dict)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off') 
    ax.set_title(f'Word Cloud for {title}')

def plot_ngrams(ngram_freq, title, ax):
    if not ngram_freq:
        ax.set_title(f'Top N-grams for {title}')
        ax.axis('off') 
        return
    
    ngrams, counts = zip(*ngram_freq)
    ax.barh(ngrams, counts, color='skyblue')
    ax.set_title(f'Top N-grams for {title}')
    ax.invert_yaxis() 
    ax.set_xlabel('Frequency')

def plot_reviews(positive_ngrams, negative_ngrams, neutral_ngrams, mask=None):
    """
    Plots the n-grams and word clouds for positive, negative, and neutral reviews.
    
    Parameters:
    - positive_ngrams: List of tuples (n-gram, frequency) for positive reviews
    - negative_ngrams: List of tuples (n-gram, frequency) for negative reviews
    - neutral_ngrams: List of tuples (n-gram, frequency) for neutral reviews
    - mask: (Optional) Mask for the word cloud shape
    """
    if positive_ngrams:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        plot_wordcloud(positive_ngrams, 'Positive Reviews', mask, axes[0])
        plot_ngrams(positive_ngrams, 'Positive Reviews', axes[1])
        plt.tight_layout()

    if negative_ngrams:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        plot_wordcloud(negative_ngrams, 'Negative Reviews', mask, axes[0])
        plot_ngrams(negative_ngrams, 'Negative Reviews', axes[1])
        plt.tight_layout()

    if neutral_ngrams:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        plot_wordcloud(neutral_ngrams, 'Neutral Reviews', mask, axes[0])
        plot_ngrams(neutral_ngrams, 'Neutral Reviews', axes[1])
        plt.tight_layout()


def get_color(sentiment_score):
    """
    Returns a color based on the sentiment score.
    Positive sentiment -> Green
    Negative sentiment -> Red
    Neutral sentiment -> Gray
    """
    if sentiment_score > 0.5:
        return "green"
    elif sentiment_score < 0:
        return "red"
    else:
        return "gray"


def clean_comments(comments):
    cleaned_comments = comments.str.strip()  
    cleaned_comments = cleaned_comments.str.replace(r'^\s*\"|\"$', '', regex=True)  
    cleaned_comments = cleaned_comments.str.replace(r'\s+', ' ', regex=True) 
    return cleaned_comments

def sentiment_by_route(domestic_data, international_data, domestic_colors, international_colors, domestic_height, international_height, year_reviews, chosen_year):
    domestic_comments = (
        year_reviews[year_reviews['is_domestic'] == True]
        .groupby('route')['header']
        .apply(lambda x: ', '.join(clean_comments(x)).strip())  
        .reset_index()
    )
    
    international_comments = (
        year_reviews[year_reviews['is_domestic'] == False]
        .groupby('route')['header']
        .apply(lambda x: ', '.join(clean_comments(x)).strip())  
        .reset_index()
    )

    # Merge comments back to sentiment data to ensure correct alignment
    domestic_mean_sentiment_with_comments = domestic_data.merge(domestic_comments, on='route', how='left')
    international_mean_sentiment_with_comments = international_data.merge(international_comments, on='route', how='left')

    # Domestic Routes Plot
    if not domestic_mean_sentiment_with_comments.empty:
        domestic_source = ColumnDataSource(data={
            'route': domestic_mean_sentiment_with_comments['route'],
            'sentiment': domestic_mean_sentiment_with_comments['vader_sentiment_numeric'],
            'color': domestic_colors,
            'comments': domestic_mean_sentiment_with_comments['header']
        })

        domestic_fig = figure(
            y_range=domestic_mean_sentiment_with_comments['route'].tolist(),
            height=int(domestic_height * 100), 
            title="Domestic Routes",
            toolbar_location=None,
            tools="",
            x_range=(-1.1, 1.1)
        )

        domestic_fig.hbar(
            y='route', 
            right='sentiment', 
            height=0.3, 
            color='color', 
            source=domestic_source
        )

        # Add HoverTool for domestic routes
        domestic_hover = HoverTool()
        domestic_hover.tooltips = [("Route", "@route"), ("Sentiment Score", "@sentiment"), ("Comments", "@comments")]
        domestic_fig.add_tools(domestic_hover)

        domestic_fig.xaxis.axis_label = "Sentiment Score"
        domestic_fig.yaxis.axis_label = "Routes"
    else:
        domestic_fig = figure(title="No Domestic Routes Data Available")

    # International Routes Plot
    if not international_mean_sentiment_with_comments.empty:
        international_source = ColumnDataSource(data={
            'route': international_mean_sentiment_with_comments['route'],
            'sentiment': international_mean_sentiment_with_comments['vader_sentiment_numeric'],
            'color': international_colors,
            'comments': international_mean_sentiment_with_comments['header']  # Use the correct comment column
        })

        international_fig = figure(
            y_range=international_mean_sentiment_with_comments['route'].tolist(),
            height=int(international_height * 100),
            title="International Routes",
            toolbar_location=None,
            tools="",
            x_range=(-1.1, 1.1)
        )

        international_fig.hbar(
            y='route', 
            right='sentiment', 
            height=0.3, 
            color='color', 
            source=international_source
        )

        # Add HoverTool for international routes
        international_hover = HoverTool()
        international_hover.tooltips = [("Route", "@route"), ("Sentiment Score", "@sentiment"), ("Comments", "@comments")]
        international_fig.add_tools(international_hover)

        international_fig.xaxis.axis_label = "Sentiment Score"
        international_fig.yaxis.axis_label = "Routes"
    else:
        international_fig = figure(title="No International Routes Data Available")

    # Combine Domestic and International Plots in a Grid
    combined_grid = gridplot([[domestic_fig, international_fig]])

    # Create a title above the plots
    title_div = Div(text=f"<h2>Routes with Best and Worst Customer Experience ({chosen_year})</h2>", width=800)

    # Combine the title and the grid in a column layout
    route_sentiments = column(title_div, combined_grid)
    
    return route_sentiments

def format_column_names(df):
    """Format column names by removing underscores and capitalizing each word."""
    df.columns = [col.replace('_', ' ').title() for col in df.columns]
    return df

def apply_filters(dataset, year_reviews, filter_widgets):
    """Apply filters to the dataset based on filter widgets' values."""
    filtered_reviews = year_reviews.copy()

    # Apply filters from the widgets
    for col, widget in filter_widgets.items():
        if isinstance(widget, pn.widgets.MultiChoice):
            selected = widget.value
            if selected:
                filtered_reviews = filtered_reviews[filtered_reviews[col].isin(selected)]
        elif isinstance(widget, pn.widgets.RangeSlider):
            selected_range = widget.value
            if selected_range:
                filtered_reviews = filtered_reviews[
                    (filtered_reviews[col] >= selected_range[0]) & 
                    (filtered_reviews[col] <= selected_range[1])
                ]

    # Update the dataset widget with the filtered data
    dataset.object = filtered_reviews  # .object for updating DataFrame widget content

def create_filter_widgets(year_reviews, exclude_columns, dataset):
    """Create filter widgets for the DataFrame columns."""
    filter_widgets = {}

    for col in year_reviews.columns:
        if col in exclude_columns:
            continue  # Skip excluded columns

        widget = None
        if year_reviews[col].dtype == 'object':
            unique_values = year_reviews[col].dropna().unique().tolist()
            widget = pn.widgets.MultiChoice(name=f'{col}', options=unique_values, sizing_mode='stretch_width')
        elif year_reviews[col].dtype in ['int64', 'float64', 'int32']:
            widget = pn.widgets.RangeSlider(
                name=f'{col}', start=year_reviews[col].min(), end=year_reviews[col].max(),
                step=1, sizing_mode='stretch_width'
            )

        if widget is not None:
            filter_widgets[col] = widget
            # Watch for changes to apply filters
            widget.param.watch(lambda event: apply_filters(dataset, year_reviews, filter_widgets), 'value')

    return filter_widgets


def get_filtered_csv(dataset):
    """Create a CSV file from the filtered dataset."""
    filtered_reviews = dataset.object  # Use .object to access the filtered DataFrame
    excluded_columns = ['Cleaned Review', 'Month']  # Replace with your actual excluded columns
    filtered_reviews_to_save = filtered_reviews.drop(columns=excluded_columns, errors='ignore')

    csv_io = io.StringIO()
    filtered_reviews_to_save.to_csv(csv_io, index=False)
    csv_io.seek(0)
    return csv_io

############################# INTERPRETATION OF RESULTS + RECOMMENDATIONS ###########################################
def result_page1(chosen_year, total_reviews, sentiment_data, is_yearly=True):
    analysis_output1 = [] 

    if is_yearly:
        analysis_output1.append("### Analyzing customer sentiment trends across all years:\n")
        
        # Trend Analysis
        sentiment_trends = {}
        years = sentiment_data.index.values
        for sentiment in sentiment_data.columns:
            counts = sentiment_data[sentiment].values
            trend = np.polyfit(years.flatten(), counts, 1)[0]
            sentiment_trends[sentiment] = trend

            if sentiment_trends[sentiment] > 0:
                analysis_output1.append(f"- **{sentiment} sentiment** shows an increasing trend over the years, reflecting improving customer sentiment or service quality.\n")
            elif sentiment_trends[sentiment] < 0:
                analysis_output1.append(f"- **{sentiment} sentiment** shows a decreasing trend, suggesting areas where service might need improvement.\n")
            else:
                analysis_output1.append(f"- **{sentiment} sentiment** remains relatively stable over time.\n")

        # Calculate percentage change
        def calculate_percentage_change(data):
            percentage_changes = {}
            for sentiment in data.columns:
                previous_year_values = data[sentiment].shift(1)
                current_year_values = data[sentiment]

                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct_change = np.where(previous_year_values == 0, np.nan, (current_year_values - previous_year_values) / previous_year_values * 100)

                percentage_changes[sentiment] = np.nanmean(pct_change)
            
            return percentage_changes

        percentage_changes = calculate_percentage_change(sentiment_data)

        analysis_output1.append("\nYearly percentage changes in sentiment:\n")
        for sentiment, change in percentage_changes.items():
            if np.isnan(change):
                analysis_output1.append(f"- **{sentiment}**: No data for the previous year to calculate change.\n")
            else:
                analysis_output1.append(f"- **{sentiment}**: {change:.2f}% change year-over-year\n")

        # Peak and Low Points Analysis
        max_sentiments = sentiment_data.max()
        min_sentiments = sentiment_data.min()
        analysis_output1.append("\nYear with highest and lowest sentiment counts:\n")
        for sentiment in sentiment_data.columns:
            max_year = sentiment_data[sentiment].idxmax()
            min_year = sentiment_data[sentiment].idxmin()
            analysis_output1.append(f"- **{sentiment} sentiment** peaked in **{max_year}** and was lowest in **{min_year}**.\n")

        # Recommendations
        analysis_output1.append("\n### Recommendations:\n")
        analysis_output1.append("Focus on enhancing strategies in years with declining sentiments and replicate successful practices from years with rising positive sentiment.\n")

    else:
        analysis_output1.append(f"## Analyzing sentiment for the year **{chosen_year}**:\n")
        
        analysis_output1.append(f"- Total reviews: A total of **{total_reviews}** reviews were submitted in **{chosen_year}**.\n")
        
        # Determine peak and lowest months
        total_reviews_per_month = sentiment_data.sum(axis=1)
        peak_month = total_reviews_per_month.idxmax()
        low_month = total_reviews_per_month.idxmin()
        
        if total_reviews_per_month[peak_month] > total_reviews_per_month.median():
            analysis_output1.append(f"- **Peak month for reviews**: **{peak_month}** had the highest number of reviews, indicating high customer activity or engagement.\n")
        else:
            analysis_output1.append(f"- **Peak month for reviews**: **{peak_month}** was within typical activity levels for the year.\n")

        analysis_output1.append(f"- **Lowest review month**: **{low_month}** had the fewest reviews. This could indicate a slower travel period or less demand.\n")

        # Analyze sentiment trends across months
        if 'Positive' in sentiment_data.columns:
            pos_peak_month = sentiment_data['Positive'].idxmax()
            pos_low_month = sentiment_data['Positive'].idxmin()
            if sentiment_data['Positive'][pos_peak_month] > sentiment_data['Positive'].median():
                analysis_output1.append(f"- **Positive sentiment**: The highest positive sentiment was recorded in **{pos_peak_month}**, indicating a period of high customer satisfaction.\n")
            else:
                analysis_output1.append(f"- **Positive sentiment**: Positive reviews were spread out with **{pos_peak_month}** seeing a relatively higher number.\n")
            
            analysis_output1.append(f"- **Lowest positive sentiment**: **{pos_low_month}** saw fewer positive reviews, which may warrant investigation into service quality during this month.\n")
        else:
            analysis_output1.append("- **Positive sentiment**: There were no or very few positive reviews in **{chosen_year}**.\n")

        if 'Negative' in sentiment_data.columns:
            neg_peak_month = sentiment_data['Negative'].idxmax()
            neg_low_month = sentiment_data['Negative'].idxmin()
            if sentiment_data['Negative'][neg_peak_month] > sentiment_data['Negative'].median():
                analysis_output1.append(f"- **Negative sentiment**: The peak in negative sentiment occurred in **{neg_peak_month}**. This suggests a challenging period for customer satisfaction.\n")
            else:
                analysis_output1.append(f"- **Negative sentiment**: Negative feedback was relatively consistent, with **{neg_peak_month}** seeing a slight uptick.\n")
            
            analysis_output1.append(f"- **Lowest negative sentiment**: **{neg_low_month}** had the least negative sentiment, indicating better customer experience during this time.\n")
        else:
            analysis_output1.append(f"- **Negative sentiment**: Negative feedback was minimal or absent for **{chosen_year}**.\n")

        # Check if the year saw more positive or negative reviews overall
        total_positive = sentiment_data['Positive'].sum()
        total_negative = sentiment_data['Negative'].sum()
        
        if total_positive > total_negative:
            analysis_output1.append(f"- **Overall sentiment**: **{chosen_year}** had a predominantly positive sentiment, with customers generally expressing satisfaction.\n")
        elif total_negative > total_positive:
            analysis_output1.append(f"- **Overall sentiment**: **{chosen_year}** saw more negative sentiment, indicating that customers were more dissatisfied during this period.\n")
        else:
            analysis_output1.append(f"- **Overall sentiment**: Sentiments were balanced, with nearly equal numbers of positive and negative reviews.\n")
        
        analysis_output1.append("\n### Recommendations:\n")
        analysis_output1.append("Investigate months with high negative sentiment to identify potential service issues. Also, replicate successful strategies from months with high positive sentiment.\n")

    return "\n".join(analysis_output1)


def result_page2(reviews_df, chosen_year):
    analysis_output2 = [] 

    # Separate domestic and international ratings
    domestic_ratings = reviews_df[reviews_df['is_domestic'] == True]
    international_ratings = reviews_df[reviews_df['is_domestic'] == False]

    # Output the year of analysis
    analysis_output2.append(f"### Analyzing customer satisfaction for domestic and international flights in **{chosen_year}**:")
    
    # Compare overall satisfaction
    domestic_overall_avg = domestic_ratings[rating_columns].mean().mean()
    international_overall_avg = international_ratings[rating_columns].mean().mean()

    if domestic_overall_avg > international_overall_avg:
        analysis_output2.append(f"\n- **Overall Satisfaction:** Domestic flights received a higher average rating ({domestic_overall_avg:.2f}) compared to international flights ({international_overall_avg:.2f}). This suggests better customer satisfaction on domestic routes.")
    elif domestic_overall_avg < international_overall_avg:
        analysis_output2.append(f"\n- **Overall Satisfaction:** International flights received a higher average rating ({international_overall_avg:.2f}) compared to domestic flights ({domestic_overall_avg:.2f}). This indicates higher customer satisfaction for international routes.")
    else:
        analysis_output2.append(f"\n- **Overall Satisfaction:** Both domestic and international flights have similar average ratings, indicating similar levels of satisfaction across both types of routes.")

    # In-depth per category comparison
    for category in rating_columns:
        domestic_avg = domestic_ratings[category].mean()
        international_avg = international_ratings[category].mean()
        
        if domestic_avg > international_avg:
            analysis_output2.append(f"- {category.replace('_', ' ').title()}: Domestic flights received higher ratings ({domestic_avg:.2f}) compared to international ({international_avg:.2f}).")
        elif domestic_avg < international_avg:
            analysis_output2.append(f"- {category.replace('_', ' ').title()}: International flights received higher ratings ({international_avg:.2f}) compared to domestic ({domestic_avg:.2f}).")
        else:
            analysis_output2.append(f"- {category.replace('_', ' ').title()}: Ratings are similar for both domestic and international flights ({domestic_avg:.2f}).")

    # Recommendations
    analysis_output2.append("\n### Recommendations: ")
    
    for category in rating_columns:
        domestic_avg = domestic_ratings[category].mean()
        international_avg = international_ratings[category].mean()
        
        if domestic_avg > international_avg:
            if category == 'seat_comfort':
                analysis_output2.append("- **Seat Comfort:** Maintain high standards for domestic flights. Consider introducing additional legroom or seat upgrades on international flights.\n Gather customer feedback on seat preferences for future improvements.")
            elif category == 'cabin_staff_service':
                analysis_output2.append("- **Cabin Staff Service:** Ensure domestic service remains top-notch. Identify key training elements from international staff that could enhance domestic service.\n Implement cross-training programs between domestic and international staff.")
            elif category == 'food_&_beverages':
                analysis_output2.append("- **Food & Beverages:** Highlight successful domestic menu items and explore their introduction on international flights.\n Conduct taste tests with customers to determine preferences.")
            elif category == 'ground_service':
                analysis_output2.append("- **Ground Service:** Evaluate domestic processes for efficiency and consider adopting successful strategies from international operations.\n Monitor customer flow during peak times and adjust staffing accordingly.")
            elif category == 'wifi_&_connectivity':
                analysis_output2.append("- **WiFi & Connectivity:** Continue to uphold high standards for domestic services while addressing any connectivity complaints on international routes.\n Invest in technology upgrades to enhance connectivity on international flights.")
            elif category == 'value_for_money':
                analysis_output2.append("- **Value for Money:** Maintain competitive pricing on domestic routes. Investigate customer perceptions of value on international flights.\n Conduct market research to better understand customer expectations regarding pricing.")
            elif category == 'inflight_entertainment':
                analysis_output2.append("- **Inflight Entertainment:** Leverage high domestic satisfaction to enhance entertainment options on international flights.\n Expand partnerships with content providers for diverse entertainment options.")
        elif domestic_avg < international_avg:
            if category == 'seat_comfort':
                analysis_output2.append("- **Seat Comfort:** Investigate the factors leading to higher ratings on international flights and apply those insights to improve domestic comfort.\n Analyze seat configuration data to find optimal layouts for customer comfort.")
            elif category == 'cabin_staff_service':
                analysis_output2.append("- **Cabin Staff Service:** Learn from international service strengths to improve domestic experiences.\n Implement best practices from high-rated international routes in domestic staff training.")
            elif category == 'food_&_beverages':
                analysis_output2.append("- **Food & Beverages:** Explore incorporating popular international menu items into domestic offerings.\n Survey frequent flyers about their preferred menu items to guide changes.")
            elif category == 'ground_service':
                analysis_output2.append("- **Ground Service:** Identify successful international practices and apply them to enhance domestic ground services.\n Regularly review customer feedback on ground service to target specific areas for improvement.")
            elif category == 'wifi_&_connectivity':
                analysis_output2.append("- **WiFi & Connectivity:** Investigate the superior performance of international services and consider upgrades for domestic offerings.\n Perform a technology audit to identify potential upgrades for domestic WiFi.")
            elif category == 'value_for_money':
                analysis_output2.append("- **Value for Money:** Focus on enhancing the perceived value of domestic offerings based on international benchmarks.\n Offer bundled services or loyalty discounts to improve perceived value.")
            elif category == 'inflight_entertainment':
                analysis_output2.append("- **Inflight Entertainment:** Enhance domestic entertainment options by adopting successful elements from international flights.\n Collect customer preferences for entertainment options to tailor offerings.")
        else:
            analysis_output2.append(f"- **{category.replace('_', ' ').title()}:** Maintain a consistent approach to service quality across both domestic and international flights while identifying specific areas for improvement.\n Regularly conduct customer satisfaction surveys to monitor performance in these areas.")

    return "\n".join(analysis_output2)


def result_page3(traveller_rating_summary, sentiment_distribution_percentage, chosen_year):
    analysis_output3 = []
    analysis_output3.append(f"### Traveler Type Analysis ({chosen_year})")
    
    # Analyzing overall rating by type of traveler
    highest_rated = traveller_rating_summary.iloc[0]
    lowest_rated = traveller_rating_summary.iloc[-1]

    analysis_output3.append(f"\n- Highest Rated Traveler Type: {highest_rated['type_of_traveller']} with an average rating of {highest_rated['mean']:.2f}.")
    analysis_output3.append(f"  This traveler type seems to have the most satisfaction on average, with {highest_rated['count']} reviews.")
    
    analysis_output3.append(f"\n- Lowest Rated Traveler Type: {lowest_rated['type_of_traveller']} with an average rating of {lowest_rated['mean']:.2f}.")
    analysis_output3.append(f"  This indicates that this group may be less satisfied compared to others, with {lowest_rated['count']} reviews.")

    # Additional insights from median rating
    analysis_output3.append(f"\n- Median Ratings: ")
    for i, row in traveller_rating_summary.iterrows():
        analysis_output3.append(f"  - {row['type_of_traveller']} has a median rating of {row['median']:.2f}.")

    # Analyze sentiment distribution by traveler type
    analysis_output3.append(f"\n ### Sentiment Analysis by Traveler Type")
    
    for traveller_type in sentiment_distribution_percentage.index:
        sentiment = sentiment_distribution_percentage.loc[traveller_type]
        
        # Identify which sentiment dominates for each traveler type
        dominant_sentiment = sentiment.idxmax()
        dominant_percentage = sentiment.max() * 100
        
        analysis_output3.append(f"\n- {traveller_type}: ")
        analysis_output3.append(f"  - Dominant Sentiment: {dominant_sentiment} ({dominant_percentage:.1f}% of total reviews)")
        
        # Analyze balance across sentiments
        positive_percentage = sentiment['Positive'] * 100 if 'Positive' in sentiment else 0
        neutral_percentage = sentiment['Neutral'] * 100 if 'Neutral' in sentiment else 0
        negative_percentage = sentiment['Negative'] * 100 if 'Negative' in sentiment else 0
        
        analysis_output3.append(f"  - Sentiment Breakdown:")
        analysis_output3.append(f"    - Positive: {positive_percentage:.1f}%")
        analysis_output3.append(f"    - Neutral: {neutral_percentage:.1f}%")
        analysis_output3.append(f"    - Negative: {negative_percentage:.1f}%")
        
        # Specific recommendations based on sentiment analysis
        if dominant_sentiment == 'Negative':
            analysis_output3.append(f"  - Recommendation: Focus on improving the experience for {traveller_type} travelers, as negative sentiment is the most prevalent.")
        elif dominant_sentiment == 'Neutral':
            analysis_output3.append(f"  - Recommendation: Consider investigating why {traveller_type} travelers are neither fully satisfied nor dissatisfied.")
        else:
            analysis_output3.append(f"  - Recommendation: Maintain the high satisfaction levels for {traveller_type} travelers, but still look for opportunities to reduce any neutral or negative experiences.")

    return "\n".join(analysis_output3)


def result_page4(seat_rating_summary, chosen_year):
    analysis_output4 = []

    analysis_output4.append(f"## Seat Type Analysis ({chosen_year})")
    
    # Analyzing overall rating by seat type
    highest_rated = seat_rating_summary.iloc[0]
    lowest_rated = seat_rating_summary.iloc[-1]

    analysis_output4.append(f"\n- Highest Rated Seat Type: {highest_rated['seat_type']} with an average rating of {highest_rated['mean']:.2f}.")
    analysis_output4.append(f"  This seat type seems to offer the most satisfaction on average, with {highest_rated['count']} reviews.")
    
    analysis_output4.append(f"\n- Lowest Rated Seat Type: {lowest_rated['seat_type']} with an average rating of {lowest_rated['mean']:.2f}.")
    analysis_output4.append(f"  This indicates that this seat type may be less satisfying compared to others, with {lowest_rated['count']} reviews.")

    # Additional insights from median rating
    analysis_output4.append(f"\n- Median Ratings: ")
    for i, row in seat_rating_summary.iterrows():
        analysis_output4.append(f"  - {row['seat_type']} has a median rating of {row['median']}.")

    # Analyze the distribution of ratings for each seat type
    for i, row in seat_rating_summary.iterrows():
        seat_type = row['seat_type']
        mean_rating = row['mean']
        median_rating = row['median']
        count_reviews = row['count']

        analysis_output4.append(f"\n- Seat Type: {seat_type}")
        analysis_output4.append(f"  - Mean Rating: {mean_rating:.2f}")
        analysis_output4.append(f"  - Median Rating: {median_rating:.2f}")
        analysis_output4.append(f"  - Number of Reviews: {count_reviews}")

        # Provide recommendations based on the ratings
        if mean_rating < 3:
            analysis_output4.append(f"  - Recommendation: Improve the features or comfort of {seat_type} seats to enhance customer satisfaction.")
        elif mean_rating >= 3 and mean_rating < 4:
            analysis_output4.append(f"  - Recommendation: {seat_type} seats are acceptable, but consider making incremental improvements to boost satisfaction.")
        else:
            analysis_output4.append(f"  - Recommendation: Continue to maintain and possibly enhance the {seat_type} seat experience, as it is highly rated.")

    return "\n".join(analysis_output4)

############################# WIDGETS & CALLBACK ###########################################
# Create buttons for the sidebar navigation
button_home = pn.widgets.Button(name="Home", button_type="light", icon='home', sizing_mode='stretch_width')
button1 = pn.widgets.Button(name="Overall Sentiment Analysis", button_type="light", icon='<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chart-line"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 19l16 0" /><path d="M4 15l4 -6l4 2l4 -5l4 4" /></svg>', sizing_mode='stretch_width')
button2 = pn.widgets.Button(name="AVG Ratings by Route Type", button_type="light", sizing_mode='stretch_width', icon='<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chart-sankey"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 3v18h18" /><path d="M3 6h18" /><path d="M3 8c10 0 8 9 18 9" /></svg>')
button3 = pn.widgets.Button(name="Sentiment by Traveler Type", button_type="light", sizing_mode='stretch_width', icon='<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chart-pie-3"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 12l-6.5 5.5" /><path d="M12 3v9h9" /><path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" /></svg>')
button4 = pn.widgets.Button(name="AVG Ratings by Seat Type", button_type="light", sizing_mode='stretch_width', icon='<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chart-bar"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 13a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v6a1 1 0 0 1 -1 1h-4a1 1 0 0 1 -1 -1z" /><path d="M15 9a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v10a1 1 0 0 1 -1 1h-4a1 1 0 0 1 -1 -1z" /><path d="M9 5a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v14a1 1 0 0 1 -1 1h-4a1 1 0 0 1 -1 -1z" /><path d="M4 20h14" /></svg>')
button5 = pn.widgets.Button(name="Wordcloud + Ngram", button_type="light", sizing_mode='stretch_width', icon='<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-language"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 5h7" /><path d="M9 3v2c0 4.418 -2.239 8 -5 8" /><path d="M5 9c0 2.144 2.952 3.908 6.7 4" /><path d="M12 20l4 -9l4 9" /><path d="M19.1 18h-6.2" /></svg>')
button6 = pn.widgets.Button(name="Sentiment by Route", button_type="light", sizing_mode='stretch_width', icon='<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-plane-inflight"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M15 11.085h5a2 2 0 1 1 0 4h-15l-3 -6h3l2 2h3l-2 -7h3l4 7z" /><path d="M3 21h18" /></svg>')
button7 = pn.widgets.Button(name="View and Save Dataset", button_type="light", sizing_mode='stretch_width', icon='<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-device-floppy"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 4h10l4 4v10a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2" /><path d="M12 14m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M14 4l0 4l-6 0l0 -4" /></svg>')

main_area = pn.Column()

def typewriter_effect(text, pane, delay=0.05):
    """
    A function that updates a Markdown pane to simulate a typewriter effect.
    """
    pane.object = "" 
    for char in text:
        pane.object += char  
        pane.param.trigger('object')  
        time.sleep(delay)

def show_page1(event=None):
    chosen_year = year_selector.value
    main_area.clear()  
    year_reviews, sentiment_counts, total_reviews = get_sentiment_analysis(reviews_df, chosen_year)
    sentiment_plots = plot_monthly_sentiment(sentiment_counts, chosen_year, total_reviews)

    grid = pn.GridSpec()
    button_analyze = pn.widgets.Button(name="Analyse", button_type="primary", button_style='outline', width=150, height=50)
    grid[0, 1:4] = pn.pane.HoloViews(sentiment_plots, align='center') 
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", sizing_mode='stretch_width') 
    grid[1, 1:4] = analysis_output_pane 

    def analyze_results1(event):
        result_text = result_page1(chosen_year, total_reviews, sentiment_counts, is_yearly=(chosen_year == 'ALL'))
        threading.Thread(target=typewriter_effect, args=(result_text, analysis_output_pane)).start()

    button_analyze.on_click(analyze_results1)
    main_area.append(grid)


def show_page2(event=None):
    chosen_year = year_selector.value 
    year_reviews, _, _ = get_sentiment_analysis(reviews_df, chosen_year)    
    main_area.clear()
    avg_ratings_plot = plot_avg_ratings(year_reviews, chosen_year)

    grid = pn.GridSpec()
    button_analyze = pn.widgets.Button(name="Analyse", button_type="primary", button_style='outline', width=150, height=50)
    grid[0,1:4] = pn.pane.HoloViews(avg_ratings_plot, align='center')
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", sizing_mode='stretch_width')
    grid[1, 1:4] = analysis_output_pane  

    def analyze_results2(event):
        result_text = result_page2(year_reviews, chosen_year)
        threading.Thread(target=typewriter_effect, args=(result_text, analysis_output_pane)).start()

    button_analyze.on_click(analyze_results2)
    main_area.append(grid)  # Add the grid to the main area



def show_page3(attr=None, old=None, new=None):
    """Display sentiment distribution by traveler type using Bokeh."""
    chosen_year = year_selector.value
    year_reviews, _, _ = get_sentiment_analysis(reviews_df, chosen_year)
    
    traveller_rating_summary = year_reviews.groupby('type_of_traveller').agg(
        mean=('rating', 'mean'),
        median=('rating', 'median'),
        count=('rating', 'size')
    ).reset_index().sort_values('mean', ascending=False)

    sentiment_distribution_percentage = year_reviews.groupby('type_of_traveller')['vader_sentiment'].value_counts(normalize=True).unstack(fill_value=0)
    main_area.clear()

    grid = pn.GridSpec()
    button_analyze = pn.widgets.Button(name="Analyse", button_type="primary", button_style='outline', width=150, height=50)
    
    grid[0, 1:4] = pn.pane.Bokeh(plot_traveller_sentiments(year_reviews, chosen_year), align='center')
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", sizing_mode='stretch_width')
    grid[1, 1:4] = analysis_output_pane  

    def analyze_results3(event):
        result_text = result_page3(traveller_rating_summary, sentiment_distribution_percentage, chosen_year)
        threading.Thread(target=typewriter_effect, args=(result_text, analysis_output_pane)).start()

    button_analyze.on_click(analyze_results3)
    main_area.append(grid)


def show_page4(event=None):
    chosen_year = year_selector.value 
    year_reviews, _, _ = get_sentiment_analysis(reviews_df, chosen_year)
    seat_rating_summary = year_reviews.groupby('seat_type')['rating'].agg(['mean', 'median', 'count']).reset_index()
    seat_rating_summary = seat_rating_summary.sort_values(by='mean', ascending=False)
    main_area.clear()

    grid = pn.GridSpec()
    button_analyze = pn.widgets.Button(name="Analyse", button_type="primary", button_style='outline', width=150, height=50)
    grid[0, 1:4] = pn.pane.HoloViews(plot_seat_type_ratings(year_reviews, chosen_year), height=600)
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", sizing_mode='stretch_width')
    grid[1, 1:4] = analysis_output_pane 

    def analyze_results4(event):
        result_text = result_page4(seat_rating_summary, chosen_year)
        threading.Thread(target=typewriter_effect, args=(result_text, analysis_output_pane)).start()

    button_analyze.on_click(analyze_results4)
    main_area.append(grid)


def show_page5(event=None):
    chosen_year = year_selector.value
    year_reviews, _, _ = get_sentiment_analysis(reviews_df, chosen_year)
    positive_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Positive']['cleaned_review']
    negative_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Negative']['cleaned_review']
    neutral_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Neutral']['cleaned_review']
    top_positive_ngrams = get_top_n_ngrams(positive_reviews, 20)
    top_negative_ngrams = get_top_n_ngrams(negative_reviews, 20)
    top_neutral_ngrams = get_top_n_ngrams(neutral_reviews, 10)
    main_area.clear()

    if top_positive_ngrams:
        plot_reviews(top_positive_ngrams, None, None, mask=mask)
        main_area.append(pn.pane.Matplotlib(plt.gcf(), height=600))

    if top_negative_ngrams:
        plot_reviews(None, top_negative_ngrams, None, mask=mask)
        main_area.append(pn.pane.Matplotlib(plt.gcf(), height=600))

    if top_neutral_ngrams:
        plot_reviews(None, None, top_neutral_ngrams, mask=mask)
        main_area.append(pn.pane.Matplotlib(plt.gcf(), height=600))

def show_page6(event=None):
    # Get the selected year from the UI
    chosen_year = year_selector.value 
    
    if chosen_year is None:
        print("No year selected")
        return

    # Perform sentiment analysis for the chosen year
    year_reviews, _, _ = get_sentiment_analysis(reviews_df, chosen_year)
    
    # Mapping vader_sentiment to numerical values
    sentiment_mapping = {'Negative': -1.0, 'Neutral': 0.5, 'Positive': 1.0} 
    year_reviews.loc[:, 'vader_sentiment_numeric'] = year_reviews['rating_sentiment'].map(sentiment_mapping)

    # Split the reviews into domestic and international routes
    domestic_routes = year_reviews[year_reviews['is_domestic'] == True]
    international_routes = year_reviews[year_reviews['is_domestic'] == False]
    
    # Calculate the mean sentiment per route for both categories
    domestic_mean_sentiment = domestic_routes.groupby('route')['vader_sentiment_numeric'].mean().reset_index()
    international_mean_sentiment = international_routes.groupby('route')['vader_sentiment_numeric'].mean().reset_index()
    
    # Sort by mean sentiment to get the best and worst routes at the top
    domestic_mean_sentiment_sorted = domestic_mean_sentiment.sort_values(by='vader_sentiment_numeric', ascending=False)
    international_mean_sentiment_sorted = international_mean_sentiment.sort_values(by='vader_sentiment_numeric', ascending=False)

    # Apply color mapping to the sentiment data
    domestic_colors = domestic_mean_sentiment_sorted['vader_sentiment_numeric'].apply(get_color).tolist()
    international_colors = international_mean_sentiment_sorted['vader_sentiment_numeric'].apply(get_color).tolist()

    # Calculate dynamic heights based on the number of routes
    base_height = 6
    height_per_route = 0.4
    fig_height_domestic = base_height + (len(domestic_mean_sentiment_sorted) * height_per_route)
    fig_height_international = base_height + (len(international_mean_sentiment_sorted) * height_per_route)
    
    # Clear the main area before rendering new content
    main_area.clear()

    # Create the sentiment charts using the improved function
    sentiment_chart = sentiment_by_route(
        domestic_mean_sentiment_sorted, 
        international_mean_sentiment_sorted, 
        domestic_colors, 
        international_colors, 
        fig_height_domestic, 
        fig_height_international,
        year_reviews,
        chosen_year
    )

    # Use a grid layout to display the sentiment chart in page 6
    grid = pn.GridSpec(sizing_mode='stretch_both')
    grid[0, 0:4] = pn.pane.Bokeh(sentiment_chart)  

    # Add the grid to the main area
    main_area.append(grid)

def show_page7(event=None):
    """Display the dataset and filter widgets in a grid layout."""
    # Assuming 'reviews_df' is your complete dataset, and 'Year' is one of the columns in the dataset
    year_reviews = reviews_df.copy()

    print(year_reviews.columns)
    print(year_reviews.dtypes)


    # Clear the main area (assuming main_area is pre-defined somewhere in your actual code)
    main_area.clear()

    # Format column names and exclude specified columns
    exclude_columns = ['Cleaned Review', 'Month', 'Vader Sentiment Numeric']  
    year_reviews = format_column_names(year_reviews)

    # Initialize a GridSpec layout
    grid = pn.GridSpec(sizing_mode='stretch_both')

    # Create a dynamic DataFrame widget to display the dataset
    dataset = pn.pane.DataFrame(year_reviews.drop(columns=exclude_columns, errors='ignore'), sizing_mode='stretch_both')

    # Create filter widgets
    filter_widgets = create_filter_widgets(year_reviews, exclude_columns, dataset)

    # Add the dataset to the grid
    grid[0, 0:4] = dataset

    # Add filter widgets to the right side of the grid
    filter_column = pn.Column(*filter_widgets.values(), sizing_mode='stretch_both')
    grid[0, 4:5] = filter_column

    # Create a FileDownload button to download the filtered dataset
    download_button = pn.widgets.FileDownload(
        filename='filtered_reviews.csv',
        callback=lambda: get_filtered_csv(dataset),
        button_type='success',
        sizing_mode='stretch_width',
        label = 'Download'
    )

    # Add the download button below the filters
    filter_column.append(download_button)

    # Append the grid to the main area
    main_area.append(grid)


def show_home_page(event=None):
    chosen_year = year_selector.value  
    main_area.clear()  
    
    grid = pn.GridSpec(sizing_mode='stretch_both')
    grid[0, 4] = pn.Column(final_display)  
    year_reviews, sentiment_counts, total_reviews = get_sentiment_analysis(reviews_df, chosen_year)
    sentiment_plots = plot_monthly_sentiment(sentiment_counts, chosen_year, total_reviews)
    grid[0, 0:4] = pn.pane.HoloViews(sentiment_plots, sizing_mode='stretch_both', align='center')  
    grid[1, 3:5] = pn.pane.HoloViews(plot_avg_ratings(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')  
    grid[1, 0:1] = pn.pane.Bokeh(plot_traveller_sentiments(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')
    grid[1, 1:3] = pn.pane.HoloViews(plot_seat_type_ratings(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')

    main_area.append(grid)



# Attach callbacks to buttons
button_home.on_click(lambda event: set_current_page('home'))
button1.on_click(lambda event: set_current_page('page1'))
button2.on_click(lambda event: set_current_page('page2'))
button3.on_click(lambda event: set_current_page('page3'))
button4.on_click(lambda event: set_current_page('page4'))
button5.on_click(lambda event: set_current_page('page5'))
button6.on_click(lambda event: set_current_page('page6'))
button7.on_click(lambda event: set_current_page('page7'))

# Set the current page and update content based on year and active page
def set_current_page(page):
    global current_page
    current_page = page
    if page == 'home':
        show_home_page()
    elif page == 'page1':
        show_page1()
    elif page == 'page2':
        show_page2()
    elif page == 'page3':
        show_page3()
    elif page == 'page4':
        show_page4()
    elif page == 'page5':
        show_page5()
    elif page == 'page6':
        show_page6()
    elif page == 'page7':
        show_page7()


# Automatically update the page content when the year selector is changed
def update_on_year_change(event):
    if current_page == 'home':
        show_home_page()
    elif current_page == 'page1':
        show_page1()
    elif current_page == 'page2':
        show_page2()
    elif current_page == 'page3':
        show_page3() 
    elif current_page == 'page4':
        show_page4()
    elif current_page == 'page5':
        show_page5()
    elif current_page == 'page6':
        show_page6()
    elif current_page == 'page7':
        show_page7()


year_selector.param.watch(update_on_year_change, 'value')


#################### SIDEBAR LAYOUT ##########################

sidebar = pn.Column(
    pn.pane.Markdown("## Menu"),
    year_selector,  
    refresh_button,
    button_home,  
    button1,
    button2,
    button3,
    button4,
    button5,
    button6,
    button7,
    styles={"width": "100%", "padding": "10px"}
)

###################### APP LAYOUT ##############################

dashboard = pn.template.BootstrapTemplate(
    title="Customer Feedback Analysis",
    sidebar=[sidebar],
    main=[alert_panel, main_area],
    header_background='black',  
    site="<b>Air New Zealand</b>",
    logo="air-nz.png",
    theme=pn.template.DarkTheme,
    sidebar_width=300,
)

# Callback to show the alert panel
def display_alert_panel(event):
    alert_panel.visible = True  # Make the alert panel visible

# Add the display alert function to the refresh button click event
refresh_button.on_click(display_alert_panel)

# Serve the Panel app
dashboard.servable()

# Initialize with the default Home page
set_current_page('home')

