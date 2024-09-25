import pandas as pd
import panel as pn
import hvplot.pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv

# Load the data
csv_file = 'nz_reviews_with_routes.csv'  
reviews_df = pd.read_csv(csv_file)

# Ensure consistent theme across Matplotlib and HvPlot
plt.style.use('Solarize_Light2')  # Use a valid Matplotlib theme
pn.extension(sizing_mode="stretch_width", theme='dark')  # Set Panel/HvPlot theme to 'dark'

# Define the plot size
PLOT_SIZE = (10, 6)

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

# Function to update the plot and recommendations when the year is selected
def update_plot(event):
    chosen_year = event.new
    year_reviews, monthly_sentiment_counts = get_sentiment_analysis(chosen_year)
    monthly_sentiment_plot.object = plot_monthly_sentiment(monthly_sentiment_counts)
    
    # Update recommendations
    recommendations = interpret_results(chosen_year, len(year_reviews), monthly_sentiment_counts, is_yearly=(chosen_year == 'ALL'))
    recommendations_pane.object = recommendations

    # Update the average ratings plot
    avg_ratings_plot.object = plot_avg_ratings(year_reviews, chosen_year)

# Visualization for monthly sentiment
def plot_monthly_sentiment(counts):
    # Define the color mapping for sentiment
    color_map = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}
    
    # Create a plot for each sentiment
    ax = counts.plot(
        kind='line',
        color=[color_map.get(col, 'black') for col in counts.columns],  # Assign colors
        marker='o',
        linewidth=2,
        figsize=(PLOT_SIZE[0], PLOT_SIZE[1])
    )

    # Customize the plot
    plt.xlabel('Month' if 'month' in counts.index.name else 'Year', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews', fontsize=14, fontweight='bold')
    plt.title('Total Reviews by Month and Sentiment' if 'month' in counts.index.name else 'Total Reviews by Year and Sentiment', fontsize=16, fontweight='bold')
    plt.legend(title='Sentiment', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Optimize layout and display the plot
    plt.tight_layout()
    return plt.gcf() 

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


# Function to generate the average ratings plot
def plot_avg_ratings(year_reviews, chosen_year):
    # Identify the rating columns
    rating_columns = ['seat_comfort', 'cabin_staff_service', 'food_&_beverages', 'ground_service', 
                      'value_for_money', 'inflight_entertainment', 'wifi_&_connectivity']
    
    # Group data by route type (Domestic/International) and calculate average ratings
    average_ratings = year_reviews.groupby(['is_domestic'])[rating_columns].mean().reset_index()
    
    # Melt the dataframe for easier plotting
    average_ratings_melted = average_ratings.melt(id_vars=['is_domestic'], 
                                                  value_vars=rating_columns,
                                                  var_name='category', 
                                                  value_name='average_rating')
    
    # Plot the average ratings comparison with consistent figure size and theme
    plt.figure(figsize=PLOT_SIZE)
    sns.barplot(data=average_ratings_melted, x='category', y='average_rating', hue='is_domestic')
    plt.title(f'Average Ratings per Category: Domestic vs International Routes ({chosen_year})', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Domestic' if label == 'True' else 'International' for label in labels], 
               title='Route Type', title_fontsize='13', fontsize='12')

    plt.tight_layout()
    return plt.gcf()  # Return the figure object for the plot


### THE DASHBOARD

# Initialize the plot with the first year in the dropdown
initial_year = 'ALL'
year_reviews, monthly_sentiment_counts = get_sentiment_analysis(initial_year)

# Initialize the plots correctly using Matplotlib pane
monthly_sentiment_plot = pn.pane.Matplotlib(plot_monthly_sentiment(monthly_sentiment_counts), sizing_mode='stretch_width')
avg_ratings_plot = pn.pane.Matplotlib(plot_avg_ratings(year_reviews, initial_year), sizing_mode='stretch_width')

# Get initial recommendations
recommendations = interpret_results(initial_year, len(year_reviews), monthly_sentiment_counts, is_yearly=(initial_year == 'ALL'))

# Create a Panel layout for recommendations
recommendations_pane = pn.pane.Markdown(recommendations)

# Add event listener to update plot and recommendations when year is selected
year_selector.param.watch(update_plot, 'value')

# Define the layout grid for the dashboard
dashboard_grid = pn.GridSpec(sizing_mode='stretch_both', max_height=800)

# Wrap year_selector in a Column with a fixed height
year_selector_column = pn.Column(year_selector, sizing_mode='fixed', height=10)  # Set the desired height

# Adding panes for plots
dashboard_grid[0, 0] = pn.Row(year_selector, sizing_mode='fixed')  # Use a Row to control sizing

dashboard_grid[1, 0] = monthly_sentiment_plot 
dashboard_grid[1, 1] = avg_ratings_plot        

# Right pane for recommendations
dashboard_grid[:, 2] = pn.Column(pn.pane.Markdown("### Recommendations"), recommendations_pane)

# Create the final layout for the dashboard
dashboard = pn.Column(dashboard_grid)

# Serve the dashboard
dashboard.show()  # This is for local development; use `dashboard.servable()` for deploying


