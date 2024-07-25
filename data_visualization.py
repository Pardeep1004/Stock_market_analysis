import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/raw_data.csv')
# Define the list of stocks
tech_list = ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS']
# Create a list of DataFrames for each company
company_list = [df[df['company_name'] == company] for company in df['company_name'].unique()]

# Plot the adjusted closing prices
def plot_adj_close(company_list, tech_list):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    
    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {tech_list[i - 1]}")
    
    plt.tight_layout()
    plt.show()

# Plot the volume
def plot_volume(company_list, tech_list):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    
    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Sales Volume for {tech_list[i - 1]}")
    
    plt.tight_layout()
    plt.show()

# Call the plotting functions
plot_adj_close(company_list, tech_list)
plot_volume(company_list, tech_list)
