import requests

# Set API Endpoint and API key
endpoint = 'live'
access_key = ' '

# Construct the API URL
url = f'http://api.coinlayer.com/api/{endpoint}?access_key={access_key}'

# Make the API request using requests library
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    exchange_rates = response.json()  # Convert the JSON response to a Python dictionary
    
    # Get user input for the currency code
    currency_code = input("Enter a currency code (e.g. BTC, ETH, etc.): ").upper()
    
    if currency_code in exchange_rates['rates']:
        rate = exchange_rates['rates'][currency_code]
        print(f"{currency_code} Exchange Rate: {rate}")
    else:
        print("Currency code not found in exchange rates.")
else:
    print("API request was not successful.")
