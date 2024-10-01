import os
import re
import pandas as pd
import yfinance as yf
import datetime
from datetime import datetime, timedelta
from ipywidgets import Button, VBox, Label, Textarea
from IPython.display import display
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np

watchlists = {


    '8/14':[
        "NVMI", "DUOL", "CVNA", "SE", "FOUR", "NTRA",
        "AZPN", "IESC", "SHOP", "CAVA", "MNDY", "BMA",
        "KRUS", "PAM", "RARE", "LRN", "MASI", "ICUI",
        "FTNT", "AXON"
    ],
    '8/15':[
        "NVMI", "APP", "LLY", "Z", "TAYD", "SHOP", "NTRA", "DUOL",
        "IESC", "NRG", "SE", "MNDY", "CAVA", "FOUR", "BMA",
        "ICUI", "PAM", "MASI", "LRN", "RARE"
    ],
    '8/16':[
        "NVMI", "APP", "LLY", "Z", "TAYD", "SHOP", "NTRA", "DUOL", "IESC", "NRG",
        "SE", "MNDY", "CAVA", "FOUR", "BMA", "ICUI", "PAM", "MASI", "LRN", "RARE"
    ],
    '8/19':["TECL", "NVDL", "COHR", "APP", "TGLS", "BMA", "SE", "RXST", "HRB", "PAM", "SKY", "KRUS", "FOUR", "EXAS", "CAVA", "RARE", "TEM", "CLMB", "SBUX", "LOAR"],
    '8/20':["TQQQ", "TECL", "FIVE", "ROM", "TREE", "APP", "ACLX", "MRVL", "COHR", "PTF", "ZG", "Z", "BMA", "RXST", "NVDL", "HALO", "PAM", "SBUX", "HRB", "SE"],
    '8/21': ["TECL", "TQQQ", "NVDL", "LITE", "APP", "ACLX", "ZG", "Z", "COHR", "NMM", "BMA", "PAM", "RXST", "FOUR", "HRB", "SE", "LOAR", "CAVA", "VKTX", "CLMB"],
    '8/22':["TARK", "TECL", "TQQQ", "NUVL", "ROKU", "NVDL", "LITE", "MRVL", "ACLX", "PTF", "ZG", "FIVE", "CRBP", "Z", "NMM", "APP", "COHR", "RXST", "NUGT", "FOUR"],
   


}
dates_watchlists = [
    {
        'date': '2024-08-14',
        'start_time': datetime(2024, 8, 14, 10, 0),
        'watchlist': watchlists['8/14'],
    },
    {
        'date': '2024-08-15',
        'start_time': datetime(2024, 8, 15, 10, 0),
        'watchlist': watchlists['8/15'],
    },
    {
        'date': '2024-08-16',
        'start_time': datetime(2024, 8, 16, 10, 0),
        'watchlist': watchlists['8/16'],
    },
    {
        'date': '2024-08-19',
        'start_time': datetime(2024, 8, 19, 10, 0),
        'watchlist': watchlists['8/19'],
    },
    {
        'date': '2024-08-20',
        'start_time': datetime(2024, 8, 20, 10, 0),
        'watchlist': watchlists['8/20'],
    },
    {
        'date': '2024-08-21',
        'start_time': datetime(2024, 8, 21, 10, 0),
        'watchlist': watchlists['8/21'],
    },
    {
        'date': '2024-08-22',
        'start_time': datetime(2024, 8, 22, 10, 0),
        'watchlist': watchlists['8/22'],
    }
]
def get_most_volatile_stocks(num_stocks=5):
    # You can change the method of getting these stocks based on your requirements
        tickers = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'NFLX', 'NVDA', 'AMD']  # Example tickers
        volatility_data = []
        
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, period='1d', interval='1m')
                if not stock_data.empty:
                    stock_data['Returns'] = stock_data['Close'].pct_change()
                    volatility = stock_data['Returns'].std()  # Calculate volatility
                    volatility_data.append((ticker, volatility))
            except Exception as e:
                print(f"Failed to download data for {ticker}: {e}")

    # Sort by volatility and return the top N stocks
        volatility_data.sort(key=lambda x: x[1], reverse=True)
        return [ticker for ticker, _ in volatility_data[:num_stocks]]

    # Update the CONFIG['watchlist'] with the most volatile stocks
def update_watchlist():
        most_volatile_stocks = get_most_volatile_stocks(num_stocks=5)
        CONFIG['watchlist'] = most_volatile_stocks
        print(f"Updated watchlist: {CONFIG['watchlist']}")

# Example CONFIG dictionary
CONFIG = {
        'watchlist': [],
        'capital': 10000,
        'test_date': datetime.now().strftime('%Y-%m-%d'),
        'start_time': datetime.now(),
        'rsi_window': 15,
        'stoch_rsi_window':15,
        'buy_rsi_threshold': 30,
        'capital': 25000,
        'target_percentage': 0.55,  
        'stop_loss_percentage': 0.5,#(-)
        'stage2tp':1,
        'stage2sl':0.5,
        'stage2':False,
        'activateS2': False,
        'reorder': True,
        'Trail': False,
        'folder_name': 'LiveDataTest',
        'clicks':370,
        'lower-limit': -375,
        'trailing-loss':200,
        'trailing-breakthrough':300,
        'trailing': False,
        'trading': True,
        "TSL": 0,
        'TSL_D': 0.55
   
        # Add other config settings as needed
    }

    # Call the update_watchlist function to refresh the watchlist
update_watchlist()
def single_day(window, rsi, watchlist, start, test, target, loss):
    CONFIG = {
        'window':window,
        'rsi': True,
        'stoch_rsi': False,
        'buy_rsi_threshold': rsi,
        'capital': 25000,
        'target_percentage': target/ 100,  
        'stop_loss_percentage': loss / 100,#(-)
        'max_time': 370,
        'stage2tp':1,
        'stage2sl':0.5,
        'stage2':False,
        'activateS2': False,
        'reorder': True,
        'Trail': False,
        'folder_name': 'LiveDataTest',
        'watchlist': watchlist,
        'start_time': start,
        'test_date': test,
        'clicks':120,
        'lower-limit': -375,
        'trailing-loss':200,
        'trailing-breakthrough':300,
        'trailing': False,
        'trading': True,
        "TSL": 0,
        'TSL_D': 0.55
       
    }
    ptime = datetime(2024, 8, 5, 9, 50)
    data_stats = {}
    def run_e():
        # Configuration Section
        profits = []
        times = []
        buys = []
       
        exits = []
        nonlocal data_stats
        data_stats = {
            'DPH': False,
            'DSLH': False,
            'Profit': 0,
            'TT': 0,
            'Succ_T': 0,
            'Failed_T': 0,
            'TTR': 0,
            'WHF': 'NA',
            'High': 0,
            'Low': 0,
            'trailing-saved':0


        }
       
        date_today = CONFIG['test_date']


        def extract_ticker_from_filename(filename):
            ticker = filename.split('_')[0]
            return ticker
        #retrieve last period minutes
        def retrieve_data(ticker, time, test_date):
            folder_path = CONFIG['folder_name']
            files = os.listdir(folder_path)
            for file in files:
               
                if extract_ticker_from_filename(file) == ticker and test_date in file:
                    data = pd.read_csv(os.path.join(folder_path, file))
                    data['Datetime'] = pd.to_datetime(data['Datetime'])
                    data['Datetime'] = data['Datetime'].dt.tz_localize(None)  # Remove timezone information
                    data = data.set_index('Datetime')
                   
                    #ctime = ctime.replace(tzinfo=None)  # Ensure the time is timezone-naive
                    data_slice = data.loc[time - pd.Timedelta(minutes=CONFIG['window']):time]  # Return last 15 minutes of data
                    return data_slice
            return pd.DataFrame()  # Return empty DataFrame if no data found
       
        def calculate_rsi(data, window=CONFIG['window']):
            delta = data['Open'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            #rs = gain / loss
            rs = gain / (loss)  # Adding a small constant to avoid division by zero
            return 100 - (100 / (1 + rs))


        def calculate_stochrsi(data, window=CONFIG['window']):
            min_val = data['Open'].rolling(window=window).min()
            max_val = data['Open'].rolling(window=window).max()
            return (data['Open'] - min_val) / (max_val - min_val)


        def check_buy(rsi):
            return rsi.iloc[-1] < CONFIG['buy_rsi_threshold']


        def buy_stock(ticker, price, capital):
            shares_bought = capital // price
            invested = price * shares_bought
            timestamp = datetime.now()
            return (f"{timestamp}: {shares_bought} shares of {ticker} purchased. "
                    f"Price bought is {price}. Total invested is {invested}."), shares_bought, invested


        def check_conditions(ticker, init_price, final_price, shares, target_price, stop_loss, time):
            profit_loss = shares * (final_price - init_price)
            if CONFIG['activateS2']:
                target2 = init_price * (1 + CONFIG['stage2tp'] / 100)
                loss2 = init_price * (1 + CONFIG['stage2sl'] / 100)
                if CONFIG['stage2'] == False and final_price >= target_price:
                    #data_stats['Succ_T'] += 1
                    CONFIG['stage2'] = True
                   
                    return (f"Target price reached at {time}: {shares} shares of {ticker} sold. "
                            f"Price sold is {final_price}. Profit/Loss is {profit_loss}."), False, profit_loss
                elif CONFIG['stage2'] == False and  final_price <= stop_loss:
                    data_stats['Failed_T'] += 1
                    return (f"Stop loss reached at {time}: {shares} shares of {ticker} sold. "
                            f"Price sold is {final_price}. Profit/Loss is {profit_loss}."), True, profit_loss
                elif CONFIG['stage2'] and final_price >= target2:
                    data_stats['Succ_T'] += 1
                    CONFIG['stage2'] = False
                    return (f"2Target price reached at {time}: {shares} shares of {ticker} sold. "
                            f"Price sold is {final_price}. Profit/Loss is {profit_loss}."), True, profit_loss
                elif CONFIG['stage2'] and final_price <= loss2:
                    data_stats['Succ_T'] += 1
                    CONFIG['stage2'] = False
                    return (f"2LTarget price reached at {time}: {shares} shares of {ticker} sold. "
                            f"Price sold is {final_price}. Profit/Loss is {profit_loss}."), True, profit_loss
                   
                else:
                    return "Check Condition Failure", False, 0
            else:
                if final_price >= target_price:
                    data_stats['Succ_T'] += 1
                   
                    return (f"Target price reached at {time}: {shares} shares of {ticker} sold. "
                            f"Price sold is {final_price}. Profit/Loss is {profit_loss}."), True, profit_loss
                elif final_price <= stop_loss:
                    data_stats['Failed_T'] += 1
                    return (f"Stop loss reached at {time}: {shares} shares of {ticker} sold. "
                            f"Price sold is {final_price}. Profit/Loss is {profit_loss}."), True, profit_loss
                else:
                    return "Check Condition Failure", False, 0




        def monitor_stock(ticker, init_price, shares, target_price, stop_loss, time):
            output = ""
            nonlocal ptime
            while True:
                data = retrieve_data(ticker, time, CONFIG['test_date'])
                if not data.empty:
                    current_price = data['Open'].iloc[-1]
                    output += (f"Current price of {ticker} at {time}: {current_price}\n"
                            f"Purchase price of {ticker}: {init_price}\n")
                    message, condition_met, profit_loss = check_conditions(ticker, init_price, current_price, shares, target_price, stop_loss, time)
                    output += message + "\n"
                    #print(ptime)
                    #print(time)
                    if time >= ptime + timedelta(minutes=CONFIG['max_time']):
                        pass
                        condition_met = True
                    if condition_met:
                        exits.append(time)
                        return output, True, profit_loss
                time += timedelta(minutes=1)
                if time > data.index.max():
                    break
            return output, False, 0




        def calculate_total_account_value(account_value, invested_stock, invested_price, invested_shares, time):
            total_value = account_value
            if invested_stock:
                data = retrieve_data(invested_stock, time, CONFIG['test_date'])
                if not data.empty:
                    current_price = data['Open'].iloc[-1]
                    total_value += invested_shares * (current_price - invested_price)
            return total_value


        def reorder_watchlist(ticker, success):
            if success:
                pass
                #NOTE: consider passing this condition to create a steadier win rate
                #nothing
                #print('Success but do not reorder')
                CONFIG['watchlist'].remove(ticker)
                CONFIG['watchlist'].append(ticker)
            else:
                pass
                CONFIG['watchlist'].remove(ticker)
                CONFIG['watchlist'].insert(0, ticker)


        def trading_loop(time, current_stock, account_value, invested_stock, invested_price, invested_shares, target_price, stop_loss):
            minute_output = ""
            if invested_stock:
                data = retrieve_data(invested_stock, time, CONFIG['test_date'])
                if len(data) < CONFIG['window']:
                    minute_output += f"Not enough data for {invested_stock} at {time}\n"
                else:
                    current_price = data['Open'].iloc[-1]
                    minute_output += f"Current price of {invested_stock} at {time}: {current_price}\n"
                    minute_output += f"Purchase price of {invested_stock}: {invested_price}\n"
                    monitor_output, condition_met, profit_loss = monitor_stock(invested_stock, invested_price, invested_shares, target_price, stop_loss, time)
                    minute_output += monitor_output
                   
                    if condition_met:
                       
                        account_value += profit_loss
                        if CONFIG['reorder']:
                            reorder_watchlist(invested_stock, profit_loss > 0)  # Reorder based on trade outcome


                        # Reset investment details
                        invested_stock = None
                        invested_price = 0
                        invested_shares = 0
                        target_price = 0
                        stop_loss = 0
                        minute_output += f"Investment exited. Searching for new buy signal...\n"
                        return minute_output, True, account_value  # Return True to indicate investment exit
                    else:
                        minute_output += f"Monitoring {invested_stock} at {time}\n"
                        return minute_output, False, account_value
            else:
                # No stock is currently invested
                minute_output += "No active investment\n"
                return minute_output, False, account_value




        def bot():
            #nonlocal buys, profits, exits
            capital = CONFIG['capital']
            account_value = capital
            ctime = CONFIG['start_time']  # Start time from config
            total_trades = 0
            test_date = CONFIG['test_date']  # Specify the date you're testing on


            # Variables to track the current investment
            invested_stock = None
            invested_price = 0
            invested_shares = 0
            target_price = 0
            stop_loss = 0
            total_value = 0


            output_widget = Textarea(value="", layout={'width': '100%', 'height': '300px'})
           
            def on_button_click(b):
                nonlocal test_date, ctime, invested_stock, invested_price, invested_shares, target_price, stop_loss, account_value, total_trades
                nonlocal profits, times
                minute_output = ""


                if invested_stock:
                    result = trading_loop(ctime, invested_stock, account_value, invested_stock, invested_price, invested_shares, target_price, stop_loss)
                    if result:
                        minute_output, exited, account_value = result
                        if exited:
                            # Increment total_trades when exiting a position
                            total_trades += 1
                            # Restart trading loop after exiting position
                            invested_stock = None
                            invested_price = 0
                            invested_shares = 0
                            target_price = 0
                            stop_loss = 0
                    else:
                        minute_output = "Error: trading_loop did not return a valid result.\n"
                else:
                    # Process new investments
                    minute_output = ""
                    for ticker in CONFIG['watchlist']:
                        minute_output += f"Processing {ticker} at {ctime}\n"
                        data = retrieve_data(ticker, ctime, test_date)
                        if len(data) < CONFIG['window']:
                            minute_output += f"Not enough data for {ticker} at {ctime}\n"
                            continue  # Skip if there's not enough data for RSI calculation
                        if CONFIG['rsi']:
                            form = calculate_rsi(data)
                        if CONFIG['stoch_rsi']:
                            form = calculate_stochrsi(data)
                        if check_buy(form ):
                            buys.append(ctime)
                            nonlocal ptime
                            ptime = ctime
                            minute_output += f"Detected buy signal for {ticker} at {ctime}\n"
                            message, shares, invested = buy_stock(ticker, data['Open'].iloc[-1], capital)
                            minute_output += message + "\n"
                            invested_stock = ticker
                            invested_price = data['Open'].iloc[-1]
                            invested_shares = shares
                            target_price = invested_price * (1 + CONFIG['target_percentage'] / 100)  # Target price percentage above purchase
                            stop_loss = invested_price * (1 - CONFIG['stop_loss_percentage'] / 100)  # Stop loss percentage below purchase
                           
                            break  # Exit loop after making an investment
                        else:
                            minute_output += f"No buy signal found for {ticker} at {ctime}\n"
               
                # Calculate and display the total account value
                total_value = calculate_total_account_value(account_value, invested_stock, invested_price, invested_shares, ctime)
               
                if (total_value >= CONFIG['capital'] * (1.01)):
                    data_stats['DPH'] = True
                    if (data_stats['DSLH'] == False):
                        data_stats['WHF'] = 'Profit'
                if (total_value <= CONFIG['capital'] * (0.99)):
                    data_stats['DSLH'] = True
                    if (data_stats['DPH'] == False):
                        data_stats['WHF'] = 'Loss'
               
                if total_trades > 0:
                    round(total_value, 2)
                    data_stats['Profit'] = total_value - CONFIG['capital']
                    data_stats['Profit'] = round(data_stats['Profit'], 2)


                    data_stats['TTR'] = data_stats['Succ_T'] / total_trades
                    data_stats['TTR'] = round(data_stats['TTR'], 3)
                    if data_stats['Profit'] > data_stats['High']:
                        data_stats['High'] = data_stats['Profit']
                    if data_stats['Profit'] < data_stats['Low']:
                        data_stats['Low'] = data_stats['Profit']
                    minute_output += f"Total account value at {ctime}: {total_value}\nEarnings: {data_stats['Profit']}\nHigh: {data_stats['High']}\nLow: {data_stats['Low']}\nDaily Profit Hit: {data_stats['DPH']}\nDaily Stop-Loss Hit: {data_stats['DSLH']}\nTotal Trades: {total_trades}\nTotal Trade Success Rate: {data_stats['TTR']}\nWHF: {data_stats['WHF']}\nSaved at: {data_stats['trailing-saved']}"
                else:
                    minute_output += f"No trades today"
                output_widget.value = minute_output
                #print(minute_output)  # Output to console
               
                profits.append(data_stats['Profit'])
                times.append(ctime)
                ctime += timedelta(minutes=1)
                time_label.value = f"Current time: {ctime.strftime('%Y-%m-%d %H:%M')}"
           
            button = Button(description="Run Bot")
            button.on_click(on_button_click)
           
            time_label = Label(value=f"Current time: {ctime.strftime('%Y-%m-%d %H:%M')}")
           
            #display(VBox([button, time_label, output_widget]))


            def automate_clicks(num_clicks):
                for _ in range(num_clicks):
                    on_button_click(None)
            count = 0
            while CONFIG['trading'] and count < CONFIG['clicks']:
                if CONFIG['trailing']:
                    if data_stats['Profit'] <= data_stats['High'] - CONFIG['trailing-loss']:
                        data_stats['trailing-saved'] = data_stats['Profit']
                        #CONFIG['trading'] = False
                        CONFIG['trailing'] = False


                if data_stats['Profit'] >= CONFIG['trailing-breakthrough']:
                    CONFIG['trailing'] = True
                if data_stats['Profit'] <= CONFIG['lower-limit']:
                    pass
                    #CONFIG['trading'] = False
                #time.sleep(60)
                #get_data_recent()
                automate_clicks(1)
                count += 1
        #open_webull()


        #get_data_recent()
        bot()        
    run_e()
    profit = data_stats.get('Profit', 0)
    return profit


def single_config(args):
    window, rsi, target, loss = args
    daily_profits = []
   
    for data_config in dates_watchlists:
        profit = single_day(window, rsi, data_config['watchlist'], data_config['start_time'], data_config['date'], target, loss)
        daily_profits.append(profit)
   
    # Calculate total profit, mean, and median
    total_profit = sum(daily_profits)
    mean_profit = np.mean(daily_profits) if daily_profits else 0
    median_profit = np.median(daily_profits) if daily_profits else 0
   
    # Format the list of daily profits for output
    daily_profits_str = ', '.join(f"{profit:.2f}" for profit in daily_profits)
   
    # Print the results in a single line
    print(f"Total Profit: {total_profit:.2f}, Mean Profit: {mean_profit:.2f}, Median Profit: {median_profit:.2f}, Window: {window}, RSI: {rsi},  T: 0.{target}, L: 0.{loss},  Daily Profits: [{daily_profits_str}]")
    return {
        'window': window,
        'rsi': rsi,
        'target': target,
        'loss': loss,
        'total_profit': total_profit,
        'mean_profit': mean_profit,
        'median_profit': median_profit,
        'daily_profits': daily_profits
    }


def run_all_permutations():
    windows = range(3, 15)
    rsis = range(10, 81, 5)
    target = range(10, 56, 5)
    loss = range(10, 56, 5)
    args = [(w, r, t, l) for w in windows for r in rsis for t in target for l in loss]
    best_config = None
    highest_median_profit = float('-inf')
    # Create a pool of worker processes
    with Pool() as pool:
        for result in pool.imap_unordered(single_config, args):
            if result['median_profit'] > highest_median_profit:
                highest_median_profit = result['median_profit']
                best_config = result
            pass
    if best_config:
        daily_profits_str = ', '.join(f"{profit:.2f}" for profit in best_config['daily_profits'])
        print(f"\nBest Configuration:")
        print(f"Window: {best_config['window']}")
        print(f"RSI: {best_config['rsi']}")
        print(f"Target: {best_config['target']}")
        print(f"Loss: {best_config['loss']}")
        print(f"Total Profit: {best_config['total_profit']:.2f}")
        print(f"Mean Profit: {best_config['mean_profit']:.2f}")
        print(f"Median Profit: {best_config['median_profit']:.2f}")
        print(f"Daily Profits: [{daily_profits_str}]")


if __name__ == "__main__":
    run_all_permutations() 


