#using coin market api

import requests

def get_cryptocurrency_price(symbol):
    url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    parameters = {
        'symbol': symbol,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': 'f2bb9bc7-842c-47ea-a2a8-295eff4115b9',
    }

    try:
        response = requests.get(url, params=parameters, headers=headers)
        data = response.json()
        print( "helooooooooooooo", data)
        if 'data' in data and symbol in data['data']:
            price = data['data'][symbol]['quote']['USD']['price']
            return price
        else:
            return None
    except (requests.ConnectionError, requests.Timeout, requests.TooManyRedirects) as e:
        print(e)
        return None

def main():
    symbol = input("Enter a cryptocurrency symbol (e.g., BTC, ETH): ").upper()
    price = get_cryptocurrency_price(symbol)
    if price is not None:
        print(f"The price of {symbol} in USD is: {price}")
    else:
        print(f"Unable to fetch price for {symbol}")

if __name__ == "__main__":
    main()
