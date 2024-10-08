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

# Callback functions that respect the selected year
import time
import threading

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

average_ratings_by_category = np.ceil(all_reviews[rating_columns].mean())

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

html_content = "<div style='font-family: Arial, sans-serif; padding: 10px;'>"
html_content += "<h3>Overall Average Ratings by Category</h3>"

for category, rating in top_5_ratings.items():
    stars = generate_star_html(rating)
    html_content += f"<div style='margin-bottom: 10px;'><strong>{category.replace('_', ' ').title()}:</strong> {stars} ({rating:.1f})</div>"

html_content += "</div>"

overall_category_ratings = pn.pane.HTML(html_content, width=500)

############################ WIDGET FOR YEAR SELECTION ############################
year_selector = pn.widgets.Select(name='Select Year', options=available_years, value='ALL', sizing_mode='stretch_width')

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
    
    average_ratings_melted = average_ratings_melted.sort_values(by=['category', 'average_rating'], ascending=[True, False])
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

mask = np.array(Image.open('/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/airplane-vector-36294843 copy.jpg'))


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


############################# INTERPRETATION OF RESULTS + RECOMMENDATIONS ###########################################
def result_page1(chosen_year, total_reviews, sentiment_data, is_yearly=True):
    analysis_output1 = [] 

    if is_yearly:
        analysis_output1.append("## Analyzing customer sentiment trends across all years:\n")
        
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
    analysis_output2.append(f"## Analyzing customer satisfaction for domestic and international flights in **{chosen_year}**:")
    
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
    analysis_output3.append(f"\n ## Sentiment Analysis by Traveler Type")
    
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
button_home = pn.widgets.Button(name="Home", button_type="primary")
button1 = pn.widgets.Button(name="Overall Sentiment Analysis", button_type="primary")
button2 = pn.widgets.Button(name="Average Ratings per Category", button_type="primary")
button3 = pn.widgets.Button(name="Sentiment by Traveler Type", button_type="primary")
button4 = pn.widgets.Button(name="Average Ratings by Seat Type", button_type="primary")
button5 = pn.widgets.Button(name="N-gram Analysis by Sentiment", button_type="primary")

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
    button_analyze = pn.widgets.Button(name="Analyse", button_type="primary", width=150, height=50)
    grid[0, 1:4] = pn.pane.HoloViews(sentiment_plots, align='center') 
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", width=900, height=300) 
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
    button_analyze = pn.widgets.Button(name="Analyze", button_type="primary", width=150, height=50)
    grid[0,1:4] = pn.pane.HoloViews(avg_ratings_plot)
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", width=900, height=300)
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
    button_analyze = pn.widgets.Button(name="Analyze", button_type="primary", width=150, height=50)
    
    grid[0, 1:4] = pn.pane.Bokeh(plot_traveller_sentiments(year_reviews, chosen_year))
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", width=900, height=300)
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
    button_analyze = pn.widgets.Button(name="Analyze", button_type="primary", width=150, height=50)
    grid[0, 1:4] = pn.pane.HoloViews(plot_seat_type_ratings(year_reviews, chosen_year), height=600)
    grid[0, 5] = button_analyze
    analysis_output_pane = pn.pane.Markdown("", width=900, height=300)
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


def show_home_page(event=None):
    chosen_year = year_selector.value  
    main_area.clear()  
    
    grid = pn.GridSpec(sizing_mode='stretch_both')
    grid[0, 2] = pn.Column(rating_display, overall_category_ratings)  
    year_reviews, sentiment_counts, total_reviews = get_sentiment_analysis(reviews_df, chosen_year)
    sentiment_plots = plot_monthly_sentiment(sentiment_counts, chosen_year, total_reviews)
    grid[0, 0:2] = pn.pane.HoloViews(sentiment_plots, sizing_mode='stretch_both', align='center')  
    grid[1, 0] = pn.pane.HoloViews(plot_avg_ratings(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')  
    grid[1, 1] = pn.pane.Bokeh(plot_traveller_sentiments(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')
    grid[1, 2] = pn.pane.HoloViews(plot_seat_type_ratings(year_reviews, chosen_year), sizing_mode='stretch_both', align='center')

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

