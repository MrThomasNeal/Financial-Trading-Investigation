import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from BSE import runBSE

def get_trading_data(replaceNames, insider):

    # Column names for the CSV file reading
    column_names = list(string.ascii_uppercase) + [f"A{x}" for x in string.ascii_uppercase]

    # Get the training data from the CSV into a DataFrame
    trading_data = pd.read_csv("bse_d000_i05_0001_avg_balance.csv", header=None, \
                               names=column_names, index_col=False)

    # If parameter replace names = True, replace the column names with the respective trader
    if replaceNames:
        newNames = {"H": "GVWY", "L": "SHVR", "P": "SNPR", "T": "ZIC", "X": "ZIP"}
        if insider:
            newNames = {"H": "GVWY", "L": "INSDR", "P": "SHVR", "T": "SNPR", "X": "ZIC", "AB": "ZIP"} # testing insider
        trading_data.rename(columns=newNames, inplace=True)

    # Fill any None values with a 0
    trading_data.fillna(0, inplace=True)

    return trading_data


class BaselinePerformanceComparison:

    def generate_graphs(self):

        # Number of times the trading session will run and be recorded
        num_runs = 10

        # Holds the average trade count for each trader
        trade_counts = []

        # Create the figure
        plt.figure(figsize=(10, 5))

        # List of traders to be tested in isolation
        traders = ["GVWY", "SHVR", "SNPR", "ZIC", "ZIP"]

        # Loop through each trader
        for trader in traders:

            # The results for each run of the simulation
            trader_results = []

            # Keep track of how many trades the trader has made
            trader_trade_count = 0

            # Loop through each run for this trader
            for _ in range(num_runs):

                # Run the BSE simulation
                runBSE(0, trader, "normal", False)
                # Get the resulting data from the CSV file
                trading_data = get_trading_data(False, False)
                # Add the data from the CSV to the trader_results array
                trader_results.append(trading_data['H'])
                # For plotting trade frequency
                trader_trade_count += len(trading_data)

            # Calculate the length of the shortest data among runs
            min_data_length = min(len(arr) for arr in trader_results)

            # Extract the data for each run up to the minimum length
            trimmed_results = [arr[:min_data_length] for arr in trader_results]

            # Convert the trimmed results list to a numpy array for easier manipulation
            trimmed_results = np.array(trimmed_results)

            # Calculate the average of the trimmed results along the first axis (axis = 0)
            average_result = np.mean(trimmed_results, axis=0)

            # Plot the average results for this trader
            plt.plot(trading_data['B'][:min_data_length], average_result, label=trader)

            # Calculate the average trade count for the trader over num_runs
            average_trade_count = trader_trade_count / num_runs

            # Append the traders average trade count to the trade_counts array
            trade_counts.append(average_trade_count)

        # Plot, label, and show graphs
        plt.xlabel("Time")
        plt.ylabel("Profits (in pence)")
        plt.title(f"Average Trader Profits In Isolation Over {num_runs} Runs")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(traders, trade_counts, color="black")
        plt.xlabel("Traders")
        plt.ylabel("Average Number Of Trades")
        plt.title(f"Average Number Of Trades Per Trader Over {num_runs} Runs")
        plt.show()


class PairwiseStrategyComparison:

    def generate_graphs(self, noise, insider, runtype):

        # Number of times to run the simulation
        num_runs = 10

        # Array of trader names which will be in the simulation
        traders = ["GVWY", "SHVR", "SNPR", "ZIC", "ZIP"]
        if insider:
            # If the insider trader is being implemented, add it to the array of trader names
            traders = ["GVWY", "SHVR", "SNPR", "ZIC", "ZIP", "INSDR"]  # When testing insider trader

        # Arrays to store the results of each trader
        GVWY_results = []
        SHVR_results = []
        SNPR_results = []
        ZIC_results = []
        ZIP_results = []
        if insider:
            INSDR_results = []  # When testing insider trader

        # Dictionary to store the trade counts of each trader for frequency testing
        trader_trade_counts = {
            "GVWY": 0,
            "SHVR": 0,
            "SNPR": 0,
            "ZIC": 0,
            "ZIP": 0
        }
        if insider:
            trader_trade_counts = {
                "GVWY": 0,
                "SHVR": 0,
                "SNPR": 0,
                "ZIC": 0,
                "ZIP": 0,
                "INSDR": 0  # when testing insider trader
            }

        # Store the best bids/offers for equilibrium testing
        equilibrium_best_bids = []
        equilibrium_best_offers = []

        # Array of array names for iterating through the arrays later in the code
        trader_array_names = ["GVWY_results", "SHVR_results", "SNPR_results", "ZIC_results", "ZIP_results"]
        if insider:
            trader_array_names = ["GVWY_results", "SHVR_results", "SNPR_results", "ZIC_results", "ZIP_results", "INSDR_results"]

        for _ in range(num_runs):
            # Run the simulation
            runBSE(noise, "normal", runtype, insider)
            # Get the data resulting from the simulation
            trading_data = get_trading_data(True, insider)

            # Loop over the traders
            for trader in traders:

                # Create the array name to store the results in the respective array
                array_name = trader + "_results"
                array = eval(array_name)
                array.append(trading_data[trader])

                # Get number of actual trades in the column and add it to the trader count
                last_value = 0
                for value in trading_data[trader]:
                    if (value != last_value):
                        trader_trade_counts[trader] += 1
                    last_value = value

            # Store the best bids/offers in the arrays
            equilibrium_best_bids.append(trading_data["C"].fillna(0))
            equilibrium_best_offers.append(trading_data["D"].fillna(0))

        # Create a graph
        plt.figure(figsize=(10, 5))

        # Average best bids and plot it on graph
        min_data_length = min(len(arr) for arr in equilibrium_best_bids)
        trimmed_results = [arr[:min_data_length] for arr in equilibrium_best_bids]
        trimmed_results = np.array(trimmed_results)
        trimmed_results = np.where(trimmed_results.astype(str) == ' None', 0, trimmed_results)
        trimmed_results = trimmed_results.astype(float)
        average_result = np.mean(trimmed_results, axis=0)
        plt.plot(trading_data["B"][:min_data_length], average_result, label="Best Bids")

        # Average best offers and plot it on graph
        min_data_length = min(len(arr) for arr in equilibrium_best_offers)
        trimmed_results = [arr[:min_data_length] for arr in equilibrium_best_offers]
        trimmed_results = np.array(trimmed_results)
        trimmed_results = np.where(trimmed_results.astype(str) == ' None', 0, trimmed_results)
        trimmed_results = trimmed_results.astype(float)
        average_result = np.mean(trimmed_results, axis=0)
        plt.plot(trading_data["B"][:min_data_length], average_result, label="Best Offers")

        # Labels and stuff for graph
        plt.xlabel("Time")
        plt.ylabel("Price (in pence)")
        plt.title(f"Best Bids/Offers Over {num_runs} Runs")
        if noise != 0:
            plt.title(f"Best Bids/Offers Over {num_runs} Runs, Noise = {noise}")
        plt.legend()
        plt.show()

        if noise != 0:
            print(f"Noise = {noise}")

        # For each array containing data for that trader from the trading sessions
        for array_name in trader_array_names:

            # Calculate the length of the shortest data among runs
            min_data_length = min(len(arr) for arr in eval(array_name))

            # Extract the data for each run up to the minimum length
            trimmed_results = [arr[:min_data_length] for arr in eval(array_name)]

            # Convert the trimmed results list to a numpy array for easier manipulation
            trimmed_results = np.array(trimmed_results)

            # Calculate the average of the trimmed results along the first axis (axis = 0)
            average_result = np.mean(trimmed_results, axis=0)

            # Plot the average results for this trader
            label_string = array_name.rstrip("_results")
            plt.plot(trading_data['B'][:min_data_length], average_result, label=label_string)

            # Get the end profit (the last profit result)
            end_profit = average_result[-1]
            print(label_string + f" = {end_profit} pence")

        # Graph stuff to show the visualisation
        plt.xlabel("Time")
        plt.ylabel("Profits (in pence)")
        plt.title(f"Trader Profits Within a Pairwise Environment Over {num_runs} Runs")
        if noise != 0:
            plt.title(f"Trader Profits Within a Pairwise Environment Over {num_runs} Runs, Noise = {noise}")
        plt.legend()
        plt.show()

        # Implement Frequency Graph

        # Array to hold the counts for each trader's trades
        trade_counts = []

        # Iterate over the traders
        for trader in traders:
            # Calculate the traders average trade count + append it to the trade_counts array
            average = trader_trade_counts[trader] / num_runs
            trade_counts.append(average)

        # Plot the graph for frequency testing
        plt.figure(figsize=(10, 5))
        plt.bar(traders, trade_counts, color="black")
        plt.xlabel("Traders")
        plt.ylabel("Average Number Of Trades")
        plt.title(f"Average Number Of Trades Per Trader Over {num_runs} Runs")
        if noise != 0:
            plt.title(f"Average Number Of Trades Per Trader Over {num_runs} Runs, Noise = {noise}")
        plt.show()


class SensitiveAnalysisAndAdaptationToChangingMarkets:

    def noise_experiments(self):

        # Different levels of noise to be used for testing
        noise_levels = [2, 5, 10, 15, 20, 25, 30, 40, 50]

        # Create pairwise strategy comparison object, used to gather data visualisation
        pairwiseStrategyComparisonObject = PairwiseStrategyComparison()

        # For each noise in the array, generate the graphs for the trading sessions to see how noise affects various
        # factors
        for level in noise_levels:
            pairwiseStrategyComparisonObject.generate_graphs(level, False, "normal")


    def volatility_experiments(self):

        # Create a pairwise strategy comparison object, used to gather data visualisation
        pairwiseStrategyComparisonObject = PairwiseStrategyComparison()

        # Different types of volatility passed in as parameters to the BSE to be used for testing
        pairwiseStrategyComparisonObject.generate_graphs(0, False, "normal")
        pairwiseStrategyComparisonObject.generate_graphs(0, False, "market_shock")
        pairwiseStrategyComparisonObject.generate_graphs(0, False, "offset")

# Baseline Performance Comparison Experiments
baselinePerformanceComparisonObject = BaselinePerformanceComparison()
baselinePerformanceComparisonObject.generate_graphs()

# Pairwise Strategy Comparison Experiments
pairwiseStrategyComparisonObject = PairwiseStrategyComparison()
pairwiseStrategyComparisonObject.generate_graphs(0, False, "normal")  # put insider to True to include it

# Sensitive Analysis and Adaptation to Changing Markets
sensitivityAnalysisAndAdaptationToChangingMarkets = SensitiveAnalysisAndAdaptationToChangingMarkets()
sensitivityAnalysisAndAdaptationToChangingMarkets.noise_experiments()
sensitivityAnalysisAndAdaptationToChangingMarkets.volatility_experiments()
