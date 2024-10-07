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
from bokeh.models import Select, ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.palettes import Category20
import numpy as np
import pandas as pd
from bokeh.transform import cumsum

# Ensure Panel is using the latest template
pn.extension()

hv.extension('bokeh')

############################ LOAD DATASETS ############################
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

rating_display = pn.pane.HTML(rating_html, width=300, height=100)

# Rating categories
rating_columns = ['seat_comfort', 'cabin_staff_service', 'food_&_beverages', 'ground_service', 
                  'wifi_&_connectivity', 'value_for_money', 'inflight_entertainment']

# Example average ratings by category (replace with actual values from your data)
average_ratings_by_category = np.ceil(reviews_df[rating_columns].mean())

top_5_ratings = average_ratings_by_category.sort_values(ascending=False).head(5)

top_5_ratings = top_5_ratings.sort_index()

# Define the star symbols (you can use CSS or images as well)
def generate_star_html(rating):
    full_star = '<span style="color: gold; font-size: 24px;">&#9733;</span>'  # Filled star (★)
    half_star = '<span style="color: gold; font-size: 24px;">&#11088;</span>'  # Half star (⯨)
    empty_star = '<span style="color: lightgray; font-size: 24px;">&#9734;</span>'  # Empty star (☆)

    stars = ""
    full_stars = int(rating)  # Full stars count
    half_stars = 1 if (rating - full_stars) >= 0.5 else 0  # Half star if the remainder is >= 0.5
    empty_stars = 5 - full_stars - half_stars  # Remaining stars are empty

    stars += full_star * full_stars  # Add full stars
    stars += half_star * half_stars  # Add half star
    stars += empty_star * empty_stars  # Add empty stars
    
    return stars

# Generate the HTML for all categories
html_content = "<div style='font-family: Arial, sans-serif; padding: 10px;'>"
html_content += "<h3>Overall Average Ratings by Category</h3>"

for category, rating in top_5_ratings.items():
    stars = generate_star_html(rating)
    # Formatting category names with stars
    html_content += f"<div style='margin-bottom: 10px;'><strong>{category.replace('_', ' ').title()}:</strong> {stars} ({rating:.1f})</div>"

html_content += "</div>"

overall_category_ratings = pn.pane.HTML(html_content, width=500)


############################ WIDGET FOR YEAR SELECTION ############################
year_selector = pn.widgets.Select(name='Select Year', options=available_years, value='ALL', sizing_mode='stretch_width')

############################ CREATE CHARTS ############################
# Variable to track current page (sentiment analysis, ratings, or home)
current_page = 'home'

# Ratings converted to numeric if they are in string format
reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')

def get_sentiment_analysis(reviews_df, chosen_year):
    if chosen_year == 'ALL':
        # Group by year for the overall data
        year_reviews = reviews_df
        sentiment_counts = year_reviews.groupby(['year', 'vader_sentiment']).size().unstack(fill_value=0)
    else:
        # Group by month for the selected year
        year_reviews = reviews_df[reviews_df['year'] == int(chosen_year)]
        sentiment_counts = year_reviews.groupby(['month', 'vader_sentiment']).size().unstack(fill_value=0)

        # Reindex by month to ensure correct order of months
        sentiment_counts = sentiment_counts.reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ], fill_value=0)

    return year_reviews, sentiment_counts

# Visualization for monthly or yearly sentiment
def plot_monthly_sentiment(sentiment_counts, chosen_year):
    sentiment_counts = sentiment_counts.reset_index()

    # Check if 'month' is in columns or 'year', and set the corresponding id_var and kdims
    time_column = 'month' if 'month' in sentiment_counts.columns else 'year'
    
    # Melt the dataframe for easier plotting
    sentiment_counts_melted = sentiment_counts.melt(
        id_vars=[time_column],  # Use time_column for id_vars
        var_name='Sentiment', 
        value_name='Count'
    )

    # Define the color mapping for the sentiments
    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}

    # Create an overlay of separate Curves for each sentiment type
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

    # Define xlabel logic first
    xlabel = 'Month' if time_column == 'month' else 'Year'

    # Apply it in both xlabel and the title
    sentiment_plots = sentiment_plots.opts(
        xlabel=xlabel,  # Use the defined xlabel
        ylabel='Number of Reviews',
        tools=[HoverTool(tooltips=[('Month/Year', '@x'), ('Count', '@y')])],
        show_legend=True,
        legend_position='top_left',
        title=f'Total Number of Reviews by {xlabel} and Sentiment ({chosen_year})',  # Use xlabel in the title as well
        height=600,
        width=900,  # Specify fixed height
        show_grid=True  # Show the grid
    )

    return sentiment_plots


# Function to generate the average ratings plot
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
    
    # Format 'category' to replace underscores with spaces and capitalize each word
    average_ratings_melted['category'] = (
        average_ratings_melted['category']
        .str.replace('_', ' ')  # Replace underscores with spaces
        .str.title()  # Capitalize each word
    )
    
    # Sort the data so that bars with higher average ratings are plotted first
    average_ratings_melted = average_ratings_melted.sort_values(by=['category', 'average_rating'], ascending=[True, False])
    
    # Define color mapping
    color_mapping = {'Domestic': '#1f77b4', 'International': '#ff7f0e'}
    
    # Ensure categories follow the original order
    average_ratings_melted['category'] = pd.Categorical(
        average_ratings_melted['category'], 
        categories=[col.replace('_', ' ').title() for col in rating_columns],  # Keep the original order
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
        ylabel='Category',  # Swap to 'ylabel'
        xlabel='Average Rating',  # Swap to 'xlabel'
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
        title=f'Average Ratings per Category:\nDomestic vs International ({chosen_year})',  # Newline between the title
        invert_axes=True  # Make the bars horizontal
    )

    return bars



def plot_traveller_sentiments(year_reviews, chosen_year):
    """
    Generate a pie chart for sentiment distribution by type of traveler using Bokeh.
    """

    # Group the data by type of traveler and sentiment
    sentiment_distribution = year_reviews.groupby(['type_of_traveller', 'vader_sentiment']).size().unstack(fill_value=0)

    # Get unique traveler types for the dropdown
    traveller_types = sentiment_distribution.index.tolist()

    # Initial pie chart setup for the first traveler type
    current_type = traveller_types[0]
    sentiments = sentiment_distribution.loc[current_type]
    data = pd.DataFrame(sentiments).reset_index()
    data.columns = ['vader_sentiment', 'value']
    data['angle'] = data['value'] / data['value'].sum() * 2 * np.pi

    # Define consistent colors for sentiments
    sentiment_color_mapping = {
        'Negative': '#FF5733',  # Custom color for Negative (red)
        'Positive': '#28A745',  # Custom color for Positive (green)
        'Neutral': '#FFC107'     # Custom color for Neutral (yellow)
    }
    
    # Assign colors based on the sentiment
    data['color'] = data['vader_sentiment'].map(sentiment_color_mapping)

    data['percentage'] = ((data['value'] / data['value'].sum()) * 100).round(2).astype(str) # Calculate percentages and format as string

    # Create a ColumnDataSource from the data
    source = ColumnDataSource(data)

    # Create the initial pie chart
    pie_chart = figure(height=450, width=450, title=current_type, toolbar_location=None,
                       tools="hover", tooltips="@vader_sentiment: @percentage%", x_range=(-0.5, 1.0))

    pie_chart.wedge(x=0, y=1, radius=0.4, 
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    line_color="white", fill_color='color', legend_field='vader_sentiment', source=source)

    pie_chart.axis.visible = False
    pie_chart.grid.grid_line_color = None

    # Create a dropdown for selecting traveler types
    dropdown = Select(title="Type of Traveler", value=current_type, options=traveller_types)

    def update(attr, old, new):
        # Update the pie chart based on the selected traveler type
        current_type = dropdown.value
        sentiments = sentiment_distribution.loc[current_type]
        data = pd.DataFrame(sentiments).reset_index()
        data.columns = ['vader_sentiment', 'value']
        data['angle'] = data['value'] / data['value'].sum() * 2 * np.pi

        # Assign colors based on the sentiment for the updated data
        data['color'] = data['vader_sentiment'].map(sentiment_color_mapping)
        
        data['percentage'] = ((data['value'] / data['value'].sum()) * 100).round(2).astype(str) # Calculate percentages and format as string

        # Update the ColumnDataSource
        source.data = {
            'vader_sentiment': data['vader_sentiment'],
            'value': data['value'],
            'angle': data['angle'],
            'color': data['color'],
            'percentage': data['percentage']
        }

        # Update pie chart title
        pie_chart.title.text = current_type

    # Attach the update function to the dropdown
    dropdown.on_change('value', update)

    # Layout the dropdown and the pie chart
    layout = column(dropdown, pie_chart)

    return layout


def plot_seat_type_ratings(year_reviews, chosen_year):
    # Group by seat type and calculate rating statistics
    seat_rating_summary = year_reviews.groupby('seat_type')['rating'].agg(['mean', 'median', 'count']).reset_index()

    # Sort the values to see which seat types have the highest/lowest ratings
    seat_rating_summary = seat_rating_summary.sort_values(by='mean', ascending=False)

    # Create a bar plot using HoloViews
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

    # Return the HoloViews plot wrapped in a Panel pane
    return bar_plot


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

# Load the airplane mask image
mask = np.array(Image.open('/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/airplane-vector-36294843 copy.jpg'))

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
    return ngram_freq.most_common(n)

# Preprocess the reviews
reviews_df['cleaned_review'] = reviews_df['review_content'].apply(preprocess_text)

# Get the top n-grams for each sentiment
positive_reviews = reviews_df[reviews_df['vader_sentiment'] == 'Positive']['cleaned_review']
negative_reviews = reviews_df[reviews_df['vader_sentiment'] == 'Negative']['cleaned_review']
neutral_reviews = reviews_df[reviews_df['vader_sentiment'] == 'Neutral']['cleaned_review']

top_positive_ngrams = get_top_n_ngrams(positive_reviews, 20)
top_negative_ngrams = get_top_n_ngrams(negative_reviews, 20)
top_neutral_ngrams = get_top_n_ngrams(neutral_reviews, 10)

# Load the airplane mask image
mask = np.array(Image.open('/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/airplane-vector-36294843 copy.jpg'))

def preprocess_ngrams(ngram_freq):
    word_list = []
    for ngram, freq in ngram_freq:
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
    
    ngrams, counts = zip(*ngram_freq)
    ax.barh(ngrams, counts, color='skyblue')
    ax.set_title(f'Top N-grams for {title}')
    ax.invert_yaxis()  # Invert y-axis to have the highest count on top
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


############################# WIDGETS & CALLBACK ###########################################
# Create buttons for the sidebar navigation
button_home = pn.widgets.Button(name="Home", button_type="primary")
button1 = pn.widgets.Button(name="Overall Sentiment Analysis", button_type="primary")
button2 = pn.widgets.Button(name="Average Ratings per Category", button_type="primary")
button3 = pn.widgets.Button(name="Sentiment by Traveler Type", button_type="primary")
button4 = pn.widgets.Button(name="Average Ratings by Seat Type", button_type="primary")
button5 = pn.widgets.Button(name="N-gram Analysis by Sentiment", button_type="primary")

# Create an area to display the content for different pages
main_area = pn.Column()

# Define the callback functions that respect the selected year
def show_page1(event=None):
    chosen_year = year_selector.value  
    main_area.clear() 
    
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    
    # Use holoviews instead of matplotlib
    sentiment_plots = plot_monthly_sentiment(sentiment_counts, chosen_year)
    main_area.append(pn.pane.HoloViews(sentiment_plots))

def show_page2(event=None):
    chosen_year = year_selector.value 
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.clear()
    avg_ratings_plot = plot_avg_ratings(year_reviews, chosen_year)
    main_area.append(pn.pane.HoloViews(avg_ratings_plot, height=600))

def show_page3(attr=None, old=None, new=None):
    """Display sentiment distribution by traveler type using Bokeh."""
    chosen_year = year_selector.value
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.clear()
    main_area.append(pn.pane.Bokeh(plot_traveller_sentiments(year_reviews, chosen_year), height=600))

def show_page4(event=None):
    chosen_year = year_selector.value 
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.clear()
    main_area.append(pn.pane.HoloViews(plot_seat_type_ratings(year_reviews, chosen_year), height=600))

def show_page5(event=None):
    # Get the selected year and sentiment analysis for that year
    chosen_year = year_selector.value
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)

    # Generate top N-grams for each sentiment (positive, negative, neutral)
    positive_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Positive']['cleaned_review']
    negative_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Negative']['cleaned_review']
    neutral_reviews = year_reviews[year_reviews['vader_sentiment'] == 'Neutral']['cleaned_review']

    top_positive_ngrams = get_top_n_ngrams(positive_reviews, 20)
    top_negative_ngrams = get_top_n_ngrams(negative_reviews, 20)
    top_neutral_ngrams = get_top_n_ngrams(neutral_reviews, 10)

    # Clear the current main area panel
    main_area.clear()

    # Plot and append the figures for positive, negative, and neutral n-grams
    if top_positive_ngrams:
        plot_reviews(top_positive_ngrams, None, None, mask=mask)  # Positive n-grams
        main_area.append(pn.pane.Matplotlib(plt.gcf(), height=600))

    if top_negative_ngrams:
        plot_reviews(None, top_negative_ngrams, None, mask=mask)  # Negative n-grams
        main_area.append(pn.pane.Matplotlib(plt.gcf(), height=600))

    if top_neutral_ngrams:
        plot_reviews(None, None, top_neutral_ngrams, mask=mask)  # Neutral n-grams
        main_area.append(pn.pane.Matplotlib(plt.gcf(), height=600))

def show_home_page(event=None):
    chosen_year = year_selector.value  
    main_area.clear()  
    
    # Create a GridSpec layout
    grid = pn.GridSpec(sizing_mode='stretch_both')

    # Add the overall rating at the top of the home page in a single cell
    grid[0, 0] = pn.Column(rating_display, overall_category_ratings)  

    # Display holoviews plots for sentiment analysis
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    sentiment_plots = plot_monthly_sentiment(sentiment_counts, chosen_year)
    
    # Set the sizing mode and alignment for each plot
    grid[0, 1:3] = pn.pane.HoloViews(sentiment_plots, sizing_mode='stretch_both', align='center')  
    grid[1, 0] = pn.pane.HoloViews(plot_avg_ratings(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')  
    grid[1, 1] = pn.pane.Bokeh(plot_traveller_sentiments(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')
    grid[1, 2] = pn.pane.HoloViews(plot_seat_type_ratings(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')

    # Append the grid layout to the main area
    main_area.append(grid)


# Attach callbacks to buttons
button_home.on_click(lambda event: set_current_page('home'))
button1.on_click(lambda event: set_current_page('page1'))
button2.on_click(lambda event: set_current_page('page2'))
button3.on_click(lambda event: set_current_page('page3'))
button4.on_click(lambda event: set_current_page('page4'))
button5.on_click(lambda event: set_current_page('page5'))


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


year_selector.param.watch(update_on_year_change, 'value')


#################### SIDEBAR LAYOUT ##########################
sidebar = pn.Column(
    pn.pane.Markdown("## Pages"),
    year_selector,  
    button_home,  
    button1,
    button2,
    button3,
    button4,
    button5,
    styles={"width": "100%", "padding": "15px"}
)

###################### APP LAYOUT ##############################
dashboard = pn.template.BootstrapTemplate(
    title="Customer Feedback Analysis",
    sidebar=[sidebar],
    main=[main_area],
    header_background="black", 
    site="Air New Zealand",
    theme=pn.template.DarkTheme,
    sidebar_width=250,
)

# Serve the Panel app
dashboard.show()

# Initialize with the default Home page
set_current_page('home')

