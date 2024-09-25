import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
from functions import get_sentiment_analysis, plot_monthly_sentiment, plot_avg_ratings, plot_sentiment_distribution, plot_avg_ratings_by_seat_type

# Load the data
csv_file = '/Users/ainna/Documents/Coding Crusade with Ainna/air-new-zealand-customer-feedback-analysis/nz_reviews_with_routes.csv'
reviews_df = pd.read_csv(csv_file)

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
monthly_sentiment_plot = pn.pane.Matplotlib(plot_monthly_sentiment(monthly_sentiment_counts), sizing_mode='stretch_width')
avg_ratings_plot = pn.pane.Matplotlib(plot_avg_ratings(year_reviews, initial_year), sizing_mode='stretch_width')
sentiment_distribution_plot = pn.pane.Matplotlib(plot_sentiment_distribution(year_reviews, initial_year), sizing_mode='stretch_width')
avg_ratings_seat_plot = pn.pane.Matplotlib(plot_avg_ratings_by_seat_type(year_reviews, initial_year), sizing_mode='stretch_width')

# Recommendations (leave blank)
recommendations_pane = pn.pane.Markdown("")

# Event listener for year selection
def update_plot(event):
    chosen_year = event.new
    year_reviews, sentiment_counts = get_sentiment_analysis(reviews_df, chosen_year)
    monthly_sentiment_plot.object = plot_monthly_sentiment(sentiment_counts)
    avg_ratings_plot.object = plot_avg_ratings(year_reviews, chosen_year)
    sentiment_distribution_plot.object = plot_sentiment_distribution(year_reviews, chosen_year)
    avg_ratings_seat_plot.object = plot_avg_ratings_by_seat_type(year_reviews, chosen_year)  # Update the new plot
    recommendations_pane.object = "No recommendations available."


year_selector.param.watch(update_plot, 'value')

# Dashboard layout
dashboard_grid = pn.GridSpec(sizing_mode='stretch_both', max_height=1000)

# Create a top row for title and year selector
top_row = pn.Row(pn.pane.Markdown("## Air New Zealand Customer Feedback Analysis", sizing_mode='stretch_width'), year_selector)

# Adjust the layout
dashboard_grid = pn.GridSpec(nrows=3, ncols=5, sizing_mode='stretch_both')

# Assign elements to the grid with specified widths
dashboard_grid[0, 0:4] = top_row  
dashboard_grid[1, 0:2] = monthly_sentiment_plot
dashboard_grid[1, 2:4] = avg_ratings_plot       
dashboard_grid[2, 0:2] = sentiment_distribution_plot 
dashboard_grid[2, 2:4] = avg_ratings_seat_plot  

# Recommendations in the third column
dashboard_grid[:, 4] = pn.Column(
    pn.pane.Markdown("### Recommendations"),
    recommendations_pane
)

# Set sizing mode if needed
dashboard_grid.sizing_mode = 'stretch_both'  # or other sizing modes as required

# Create and serve the dashboard
dashboard = pn.Column(dashboard_grid)
dashboard.show()
