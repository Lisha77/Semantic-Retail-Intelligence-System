import pandas as pd
import numpy as np
import datetime

def generate_data(num_rows=1000):
    np.random.seed(42)
    
    branches = ['A', 'B', 'C']
    cities = {'A': 'Yangon', 'B': 'Mandalay', 'C': 'Naypyitaw'}
    customer_types = ['Member', 'Normal']
    gender = ['Female', 'Male']
    product_lines = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 
                     'Sports and travel', 'Food and beverages', 'Fashion accessories']
    payment_methods = ['Cash', 'Credit card', 'Ewallet']

    data = {
        'Invoice ID': [f'{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}' for _ in range(num_rows)],
        'Branch': np.random.choice(branches, num_rows),
        'Customer type': np.random.choice(customer_types, num_rows),
        'Gender': np.random.choice(gender, num_rows),
        'Product line': np.random.choice(product_lines, num_rows),
        'Unit price': np.round(np.random.uniform(10, 100, num_rows), 2),
        'Quantity': np.random.randint(1, 11, num_rows),
        'Tax 5%': 0.0,
        'Total': 0.0,
        'Date': [datetime.date(2023, 1, 1) + datetime.timedelta(days=np.random.randint(0, 90)) for _ in range(num_rows)],
        'Time': [f'{np.random.randint(10, 21)}:{np.random.randint(10, 59)}' for _ in range(num_rows)],
        'Payment': np.random.choice(payment_methods, num_rows),
        'COGS': 0.0,
        'Gross margin percentage': 4.76190476,
        'Gross income': 0.0,
        'Rating': np.round(np.random.uniform(4, 10, num_rows), 1)
    }

    df = pd.DataFrame(data)
    df['Tax 5%'] = 0.05 * df['Unit price'] * df['Quantity']
    df['Total'] = (df['Unit price'] * df['Quantity']) + df['Tax 5%']
    df['COGS'] = df['Unit price'] * df['Quantity']
    df['Gross income'] = df['Total'] - df['COGS']
    df['City'] = df['Branch'].map(cities)
    
    # Reorder columns to match Kaggle format roughly
    cols = ['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender', 'Product line', 
            'Unit price', 'Quantity', 'Tax 5%', 'Total', 'Date', 'Time', 'Payment', 
            'COGS', 'Gross margin percentage', 'Gross income', 'Rating']
    df = df[cols]
    
    df.to_csv('supermarket_sales.csv', index=False)
    print("supermarket_sales.csv generated successfully!")

if __name__ == "__main__":
    generate_data()
