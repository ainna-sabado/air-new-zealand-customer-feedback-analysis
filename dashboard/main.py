import panel as pn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load wine dataset from sklearn
def load_dataset():
    wine = datasets.load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df["WineType"] = [wine.target_names[typ] for typ in wine.target]
    return wine, wine_df

wine, wine_df = load_dataset()

# Initialize Panel extension
pn.extension()

# Sidebar - Filter example
sidebar = pn.widgets.Select(name='Wine Type', options=wine_df['WineType'].unique().tolist(), value='class_0')

# Define plots (Seaborn + Matplotlib)
def plot1():
    plt.figure(figsize=(4, 3))
    sns.histplot(wine_df['alcohol'], kde=True)
    plt.title("Alcohol Distribution")
    return plt.gcf()

def plot2():
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='WineType', y='malic_acid', data=wine_df)
    plt.title("Malic Acid by Wine Type")
    return plt.gcf()

def plot3():
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='ash', y='alcohol', hue='WineType', data=wine_df)
    plt.title("Ash vs Alcohol")
    return plt.gcf()

def plot4():
    plt.figure(figsize=(4, 3))
    sns.histplot(wine_df['alcalinity_of_ash'], kde=True)
    plt.title("Alcalinity of Ash Distribution")
    return plt.gcf()

def plot5():
    plt.figure(figsize=(4, 3))
    sns.violinplot(x='WineType', y='magnesium', data=wine_df)
    plt.title("Magnesium by Wine Type")
    return plt.gcf()

def plot6():
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='total_phenols', y='alcohol', hue='WineType', data=wine_df)
    plt.title("Total Phenols vs Alcohol")
    return plt.gcf()

def plot7():
    plt.figure(figsize=(4, 3))
    sns.histplot(wine_df['flavanoids'], kde=True)
    plt.title("Flavanoids Distribution")
    return plt.gcf()

def plot8():
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='nonflavanoid_phenols', y='proline', hue='WineType', data=wine_df)
    plt.title("Nonflavanoid Phenols vs Proline")
    return plt.gcf()

def plot9():
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='hue', y='color_intensity', hue='WineType', data=wine_df)
    plt.title("Hue vs Color Intensity")
    return plt.gcf()

# Organizing the plots into 3 rows and 3 columns
main_area = pn.GridSpec(sizing_mode='stretch_both')

main_area[0, 0] = pn.pane.Matplotlib(plot1(), sizing_mode='stretch_both')
main_area[0, 1] = pn.pane.Matplotlib(plot2(), sizing_mode='stretch_both')
main_area[0, 2] = pn.pane.Matplotlib(plot3(), sizing_mode='stretch_both')

main_area[1, 0] = pn.pane.Matplotlib(plot4(), sizing_mode='stretch_both')
main_area[1, 1] = pn.pane.Matplotlib(plot5(), sizing_mode='stretch_both')
main_area[1, 2] = pn.pane.Matplotlib(plot6(), sizing_mode='stretch_both')

main_area[2, 0] = pn.pane.Matplotlib(plot7(), sizing_mode='stretch_both')
main_area[2, 1] = pn.pane.Matplotlib(plot8(), sizing_mode='stretch_both')
main_area[2, 2] = pn.pane.Matplotlib(plot9(), sizing_mode='stretch_both')

# Combine sidebar and main area in a layout
dashboard = pn.Row(
    pn.Column(sidebar, width=200),  # Sidebar
    main_area  # Main grid layout
)

# Show the dashboard
dashboard.show()
