import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Ensure Panel is using the latest template
pn.extension()

########################### LOAD DATASET ##################
# Set the CSV file path and year for analysis
csv_file = '/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/nz_reviews_with_routes.csv'

# Load the dataset
reviews_df = pd.read_csv(csv_file)

# Convert date column to datetime and extract the year
reviews_df['date'] = pd.to_datetime(reviews_df['date'], format='%Y-%m-%d')
reviews_df['year'] = reviews_df['date'].dt.year

# Extract the month names
reviews_df['month'] = reviews_df['date'].dt.strftime('%B')

# Available years for dropdown
available_years = [str(year) for year in sorted(reviews_df['year'].unique())] + ['ALL']

# Define plot size
PLOT_SIZE = (10, 6)

############################ OVERALL RESULTS ############################
all_reviews = "air_nz_cleaned_data.csv"
all_reviews = pd.read_csv(all_reviews)

# Overall average rating
average_rating = all_reviews['rating'].mean()

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
average_ratings_by_category = np.ceil(all_reviews[rating_columns].mean())

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
def plot_monthly_sentiment(sentiment_counts, year_reviews, chosen_year):
    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}
    
    ax = sentiment_counts.plot(
        kind='line',
        color=[color_map.get(col, 'black') for col in sentiment_counts.columns],
        marker='o',
        linewidth=2,
        figsize=PLOT_SIZE
    )
    
    # Labeling based on whether the index is months or years
    x_label = 'Month' if 'month' in sentiment_counts.index.names else 'Year'
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    
    plt.ylabel('Number of Reviews', fontsize=14, fontweight='bold')
    
    # Title changes based on the time period
    title = f'Total Reviews ({len(year_reviews)}) by {x_label} and Sentiment ({chosen_year})'
    plt.title(title, fontsize=16, fontweight='bold')
    
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

def plot_traveller_sentiments(year_reviews, chosen_year):
    """
    Generate pie charts for sentiment distribution by type of traveler.
    """
    # Define color mapping for sentiments
    COLOR_MAP = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}

    # Group the data by type of traveler and sentiment
    sentiment_distribution = year_reviews.groupby(['type_of_traveller', 'vader_sentiment']).size().unstack(fill_value=0)
    traveller_types = sentiment_distribution.index

    # Create subplots for each type of traveler (2 rows, 2 columns)
    num_travellers = len(traveller_types)
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Iterate through each type of traveler and create a pie chart
    for i, traveller_type in enumerate(traveller_types):
        sentiments = sentiment_distribution.loc[traveller_type]
        axes[i].pie(
            sentiments,
            labels=sentiments.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=[COLOR_MAP.get(sentiment, 'grey') for sentiment in sentiments.index]  # Use color mapping
        )
        axes[i].set_title(traveller_type)

    # Hide any unused subplots if there are fewer than 4 types of travelers
    for j in range(num_travellers, rows * cols):
        fig.delaxes(axes[j])

    # Adjust layout for titles and overall title
    plt.subplots_adjust(top=0.9)  # Adjust the top space for the title
    plt.suptitle(f'Sentiment Distribution for Different Types of Travelers ({chosen_year})', fontsize=16, fontweight='bold')
    return plt.gcf()

def plot_seat_type_ratings(year_reviews, chosen_year):
    # Group by seat type and calculate rating statistics
    seat_rating_summary = year_reviews.groupby('seat_type')['rating'].agg(['mean', 'median', 'count']).reset_index()

    # Sort the values to see which seat types have the highest/lowest ratings
    seat_rating_summary = seat_rating_summary.sort_values(by='mean', ascending=False)

    # Set the figure size for better readability
    plt.figure(figsize=PLOT_SIZE)

    # Create a bar plot to visualize the mean ratings by seat type
    sns.barplot(x='seat_type', y='mean', data=seat_rating_summary, palette='coolwarm')

    # Add labels and title
    plt.title(f'Average Ratings by Seat Type ({chosen_year})')
    plt.xlabel('Seat Type')
    plt.ylabel('Average Rating')

    # Display the plot
    plt.tight_layout()
    return plt.gcf()


############################# WIDGETS & CALLBACK ###########################################
# Create buttons for the sidebar navigation
button_home = pn.widgets.Button(name="Home", button_type="primary")
button1 = pn.widgets.Button(name="Overall Sentiment Analysis", button_type="primary")
button2 = pn.widgets.Button(name="Average Ratings per Category", button_type="primary")
button3 = pn.widgets.Button(name="Sentiment by Traveler Type", button_type="primary")
button4 = pn.widgets.Button(name="Average Ratings by Seat Type", button_type="primary")

# Create an area to display the content for different pages
main_area = pn.Column()

# Define the callback functions that respect the selected year
def show_page1(event=None):
    chosen_year = year_selector.value  # Get the selected year from the widget
    main_area.clear()  # Clear previous content
    
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.append(pn.pane.Matplotlib(plot_monthly_sentiment(sentiment_counts, year_reviews, chosen_year), height=600))  # Pass year_reviews and chosen_year

def show_page2(event=None):
    chosen_year = year_selector.value  # Get the selected year from the widget
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.clear()
    main_area.append(pn.pane.Matplotlib(plot_avg_ratings(year_reviews, chosen_year), height=600))  # Plot average ratings

def show_page3(event=None):
    """Display sentiment distribution by traveler type."""
    chosen_year = year_selector.value  # Get the selected year from the widget
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.clear()
    main_area.append(pn.pane.Matplotlib(plot_traveller_sentiments(year_reviews, chosen_year), height=600))

def show_page4(event=None):
    chosen_year = year_selector.value  # Get the selected year from the widget
    year_reviews, _ = get_sentiment_analysis(reviews_df, chosen_year)  # Get reviews for the selected year
    main_area.clear()
    main_area.append(pn.pane.Matplotlib(plot_seat_type_ratings(year_reviews, chosen_year), height=600))  # Plot average ratings by seat type

def show_home_page(event=None):
    chosen_year = year_selector.value  # Get the selected year from the widget
    main_area.clear()  # Clear previous content
    
    # Add the overall rating at the top of the home page
    main_area.append(rating_display)  # Display overall rating
    main_area.append(overall_category_ratings)

    # Display plots for sentiment, average ratings, traveler sentiments, and seat type ratings
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    main_area.append(pn.pane.Matplotlib(plot_monthly_sentiment(sentiment_counts, year_reviews, chosen_year), height=400))  # Pass year_reviews and chosen_year
    main_area.append(pn.pane.Matplotlib(plot_avg_ratings(year_reviews, chosen_year), height=400))  # Plot average ratings
    main_area.append(pn.pane.Matplotlib(plot_traveller_sentiments(year_reviews, chosen_year), height=600))  # Plot sentiment by traveler type
    main_area.append(pn.pane.Matplotlib(plot_seat_type_ratings(year_reviews, chosen_year), height=600))  # Plot average ratings by seat type

# Attach callbacks to buttons
button_home.on_click(lambda event: set_current_page('home'))
button1.on_click(lambda event: set_current_page('page1'))
button2.on_click(lambda event: set_current_page('page2'))
button3.on_click(lambda event: set_current_page('page3'))
button4.on_click(lambda event: set_current_page('page4'))

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

year_selector.param.watch(update_on_year_change, 'value')

#################### SIDEBAR LAYOUT ##########################
sidebar = pn.Column(
    pn.pane.Markdown("## Pages"),
    year_selector,  # Add year selector to the sidebar
    button_home,  # Add Home button to the sidebar
    button1,
    button2,
    button3,
    button4,
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
dashboard.servable()

# Initialize with the default Home page
set_current_page('home')

