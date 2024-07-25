import yfinance as yf
import pandas as pd
from datetime import datetime

# Define the list of stocks
tech_list = ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS']

# Set the date range for data collection (last 1 year)
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download the stock data
def download_stock_data(stock_list, start_date, end_date):
    stock_data = {}
    for stock in stock_list:
        stock_data[stock] = yf.download(stock, start=start_date, end=end_date)
    return stock_data

stock_data = download_stock_data(tech_list, start, end)

# Add company names to the data
company_list = [stock_data[stock] for stock in tech_list]
company_name = ["HDFC Bank Limited", "ICICI Bank Limited", "Axis Bank Limited", "Kotak Mahindra Bank Limited"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

# Concatenate all data into a single DataFrame
df = pd.concat(company_list, axis=0)

# Save the data to a CSV file (optional)
df.to_csv('data/raw_data.csv', index=False)

# Display the first few rows of the DataFrame
print(df.head(10))
