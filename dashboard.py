import pandas as pd
import panel as pn
import hvplot.pandas
import numpy as np

# Load the data
csv_file = 'nz_reviews_with_routes.csv'  
reviews_df = pd.read_csv(csv_file)

def parse_date(date_flown_str):
    return pd.to_datetime(date_flown_str, format='%Y-%m-%d')

# Apply the function to the 'date_flown' column
reviews_df['date'] = reviews_df['date'].apply(parse_date)  # Ensure correct column name

# Extract month and year
reviews_df['month'] = reviews_df['date'].dt.month_name()
reviews_df['year'] = reviews_df['date'].dt.year

# Get available years for the dropdown menu
available_years = sorted(reviews_df['year'].unique())  # Simplified to only unique years
available_years = [str(year) for year in available_years] + ['ALL']

# Create a Panel dropdown for year selection
year_selector = pn.widgets.Select(name='Select Year', options=available_years, value='ALL')

# Function to get yearly or monthly sentiment analysis
def get_sentiment_analysis(chosen_year):
    if chosen_year == 'ALL':
        year_reviews = reviews_df
        sentiment_counts = year_reviews.groupby(['year', 'vader_sentiment']).size().unstack(fill_value=0)
    else:
        year_reviews = reviews_df[reviews_df['year'] == int(chosen_year)]  # Ensure correct type comparison
        sentiment_counts = year_reviews.groupby(['month', 'vader_sentiment']).size().unstack(fill_value=0)

    # Sort months in calendar order
    if chosen_year != 'ALL':
        sentiment_counts = sentiment_counts.reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])

    return year_reviews, sentiment_counts

# Generate the sentiment plot based on user selection
def update_plot(event):
    chosen_year = event.new
    year_reviews, monthly_sentiment_counts = get_sentiment_analysis(chosen_year)
    monthly_sentiment_plot.object = plot_monthly_sentiment(monthly_sentiment_counts)
    
    # Update recommendations
    recommendations = interpret_results(chosen_year, len(year_reviews), monthly_sentiment_counts, is_yearly=(chosen_year == 'ALL'))
    recommendations_pane.object = recommendations

# Visualization for monthly sentiment
def plot_monthly_sentiment(counts):
    # Define the color mapping for sentiment
    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}

    # Create a line plot for each sentiment separately
    plot = counts.hvplot.line(
        title='Total Reviews by Month and Sentiment',
        xlabel='Month',
        ylabel='Number of Reviews',
        color=list(color_map.values()),  # Assign colors based on the order of sentiment
    )
    return plot

# Interpretation function for recommendations
def interpret_results(chosen_year, total_reviews, sentiment_data, is_yearly=True):
    recommendations = []

    if is_yearly:
        recommendations.append("Analyzing customer sentiment trends across all years:")
        
        # Trend Analysis
        sentiment_trends = {}
        years = sentiment_data.index.values
        for sentiment in sentiment_data.columns:
            counts = sentiment_data[sentiment].values
            trend = np.polyfit(years.flatten(), counts, 1)[0]
            sentiment_trends[sentiment] = trend

            if sentiment_trends[sentiment] > 0:
                recommendations.append(f"{sentiment} sentiment shows an increasing trend over the years, reflecting improving customer sentiment or service quality.")
            elif sentiment_trends[sentiment] < 0:
                recommendations.append(f"{sentiment} sentiment shows a decreasing trend, suggesting areas where service might need improvement.")
            else:
                recommendations.append(f"{sentiment} sentiment remains relatively stable over time.")

        # Percentage change calculation
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

        # Call the modified function for percentage changes
        percentage_changes = calculate_percentage_change(sentiment_data)

        recommendations.append("Yearly percentage changes in sentiment:")
        for sentiment, change in percentage_changes.items():
            if np.isnan(change):
                recommendations.append(f"{sentiment}: No data for the previous year to calculate change.")
            else:
                recommendations.append(f"{sentiment}: {change:.2f}% change year-over-year.")

        # Peak and Low Points Analysis
        max_sentiments = sentiment_data.max()
        min_sentiments = sentiment_data.min()
        recommendations.append("Year with highest and lowest sentiment counts:")
        for sentiment in sentiment_data.columns:
            max_year = sentiment_data[sentiment].idxmax()
            min_year = sentiment_data[sentiment].idxmin()
            recommendations.append(f"{sentiment} sentiment peaked in {max_year} and was lowest in {min_year}.")

        recommendations.append("Recommendation: Focus on enhancing strategies in years with declining sentiments and replicate successful practices from years with rising positive sentiment.")

    else:
        recommendations.append(f"Analyzing sentiment for the year {chosen_year}:")
        
        recommendations.append(f"Total reviews: A total of {total_reviews} reviews were submitted in {chosen_year}.")
        
        # Determine peak and lowest months
        total_reviews_per_month = sentiment_data.sum(axis=1)
        peak_month = total_reviews_per_month.idxmax()
        low_month = total_reviews_per_month.idxmin()
        
        recommendations.append(f"Peak month for reviews: {peak_month} had the highest number of reviews, indicating high customer activity or engagement.")
        recommendations.append(f"Lowest review month: {low_month} had the fewest reviews. This could indicate a slower travel period or less demand.")

        # Analyze sentiment trends across months
        if 'Positive' in sentiment_data.columns:
            pos_peak_month = sentiment_data['Positive'].idxmax()
            pos_low_month = sentiment_data['Positive'].idxmin()
            recommendations.append(f"Positive sentiment: The highest positive sentiment was recorded in {pos_peak_month}, indicating a period of high customer satisfaction.")
            recommendations.append(f"Lowest positive sentiment: {pos_low_month} saw fewer positive reviews, which may warrant investigation into service quality during this month.")
        else:
            recommendations.append("Positive sentiment: There were no or very few positive reviews in {chosen_year}.")

        if 'Negative' in sentiment_data.columns:
            neg_peak_month = sentiment_data['Negative'].idxmax()
            neg_low_month = sentiment_data['Negative'].idxmin()
            recommendations.append(f"Negative sentiment: The peak in negative sentiment occurred in {neg_peak_month}. This suggests a challenging period for customer satisfaction.")
            recommendations.append(f"Lowest negative sentiment: {neg_low_month} had the least negative sentiment, indicating better customer experience during this time.")
        else:
            recommendations.append("Negative sentiment: Negative feedback was minimal or absent for {chosen_year}.")

        # Check if the year saw more positive or negative reviews overall
        total_positive = sentiment_data['Positive'].sum()
        total_negative = sentiment_data['Negative'].sum()
        
        if total_positive > total_negative:
            recommendations.append(f"Overall sentiment: {chosen_year} had a predominantly positive sentiment, with customers generally expressing satisfaction.")
        elif total_negative > total_positive:
            recommendations.append(f"Overall sentiment: {chosen_year} saw more negative sentiment, indicating that customers were more dissatisfied during this period.")
        else:
            recommendations.append(f"Overall sentiment: Sentiments were balanced, with nearly equal numbers of positive and negative reviews.")
        
        recommendations.append("Recommendation: Investigate months with high negative sentiment to identify potential service issues. Also, replicate successful strategies from months with high positive sentiment.")

    return "\n".join(recommendations)

# Initialize the plot with the first year in the dropdown
initial_year = available_years[0]
year_reviews, monthly_sentiment_counts = get_sentiment_analysis(initial_year)
monthly_sentiment_plot = plot_monthly_sentiment(monthly_sentiment_counts)

# Get initial recommendations
recommendations = interpret_results(initial_year, len(year_reviews), monthly_sentiment_counts, is_yearly=(initial_year == 'ALL'))

# Create a Panel layout
pn.extension()

# Create the dashboard layout
monthly_sentiment_plot = pn.pane.HoloViews(monthly_sentiment_plot)  # Ensure the plot can be updated
recommendations_pane = pn.pane.Markdown(recommendations)  # Pane for recommendations

dashboard = pn.Row(
    pn.Column(year_selector, monthly_sentiment_plot),  # Left pane for the plot and year selector
    pn.Column(pn.pane.Markdown("### Recommendations"), recommendations_pane)  # Right pane for recommendations
)

# Add event listener to update plot and recommendations when year is selected
year_selector.param.watch(update_plot, 'value')

# Show the dashboard
dashboard.servable()
