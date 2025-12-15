# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

def generate_plot(csv_filepath, output_filename="payoff_vs_noise_plot.png"):
    """
    Generates a plot of average payoff vs. noise (ea/ep) from CSV data.

    Args:
        csv_filepath (str): Path to the input CSV file.
        output_filename (str): Filename for the saved plot.
    """
    # --- Load Data ---
    try:
        df = pd.read_csv(csv_filepath)
        print(f"Successfully loaded data from: {csv_filepath}")
        # Debugging: print column names and head
        # print("Column headers:", df.columns.tolist())
        # print("First 5 rows:\n", df.head())
    except FileNotFoundError:
        print(f"Error: File not found: {csv_filepath}", file=sys.stderr)
        print("Ensure 'results.csv' is in the script's directory.", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {csv_filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Check Required Columns ---
    required_columns = ['model', 'ea', 'ep', 'swap_prob', 'avg_reward']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV '{csv_filepath}' must contain columns: {required_columns}", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # --- Filter Data ---
    # Assume we only care about cases where ea == ep for the x-axis
    df_filtered = df[df['ea'] == df['ep']].copy()
    if df_filtered.empty:
        print("Warning: No rows found where 'ea' == 'ep'. Plot might be empty.", file=sys.stderr)
        # Option: Plot against 'ea' only, without filtering
        # df_filtered = df.copy()
        # x_axis_column = 'ea'
        # x_axis_label = 'ea Value'
        x_axis_column = 'ea' # Still use 'ea' but acknowledge it might be empty
        x_axis_label = 'ea / ep Value (filtered where ea==ep)'
    else:
        # Use 'ea' as the representative noise value on the X-axis
        x_axis_column = 'ea'
        x_axis_label = 'ea / ep Value'
        print(f"Found {len(df_filtered)} rows where ea == ep.")


    # --- Prepare Data for Plotting ---
    df_model1 = df_filtered[df_filtered['model'] == 1].sort_values(x_axis_column)
    df_model2 = df_filtered[df_filtered['model'] == 2].sort_values(x_axis_column)

    swap_probabilities = df_model2['swap_prob'].unique()
    swap_probabilities.sort() # Sort for consistent plot order

    print(f"Data found for Model 1: {'Yes' if not df_model1.empty else 'No'}")
    print(f"Data found for Model 2: {'Yes' if not df_model2.empty else 'No'}")
    if not df_model2.empty:
        print(f"Unique swap_prob values for Model 2: {swap_probabilities}")

    # --- Create Plot ---
    plt.style.use('seaborn-whitegrid') # Use a seaborn style
    fig, ax = plt.subplots(figsize=(10, 6)) # Figure size

    # --- Plot Model 1 ---
    if not df_model1.empty:
        ax.plot(df_model1[x_axis_column], df_model1['avg_reward'],
                label='Model 1 (Global)', # Legend label
                linewidth=2,
                linestyle='--',     # Linestyle
                color='black',      # Line color
                marker='^',         # Marker style
                markersize=5)       # Marker size
    else:
        print("No data to plot for Model 1.")

    # --- Plot Model 2 (per swap_prob) ---
    # Define styles for distinguishing curves
    markers = ['o', 's', 'D', 'v', 'X', 'P', '*']
    linestyles = ['-', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5))]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(swap_probabilities))) # Colormap

    if not df_model2.empty:
        for i, swap_prob in enumerate(swap_probabilities):
            df_swap = df_model2[df_model2['swap_prob'] == swap_prob]
            if not df_swap.empty:
                marker_style = markers[i % len(markers)]
                line_style = linestyles[i % len(linestyles)]
                color = colors[i % len(colors)]
                ax.plot(df_swap[x_axis_column], df_swap['avg_reward'],
                        label=f'Model 2 (swap={swap_prob:.2f})', # Label with swap_prob
                        linewidth=2,
                        linestyle=line_style,
                        marker=marker_style,
                        markersize=5,
                        color=color)
            else:
                # This case might occur if filtering removed all data for a specific swap_prob
                 print(f"No data to plot for Model 2 with swap_prob={swap_prob} after filtering.")
    elif not df_filtered.empty: # Only print if we expected data
        print("No data to plot for Model 2 (after filtering).")

    # --- Axis Settings and Title ---
    ax.set_title('Average Payoff vs. Noise Level (ea/ep)', fontsize=14) # Plot title
    ax.set_xlabel(x_axis_label, fontsize=12) # X-axis label
    ax.set_ylabel('Average Payoff (avg_reward)', fontsize=12) # Y-axis label

    # Set axis ticks based on desired range
    x_ticks = np.arange(0.0, 0.2 + 0.05, 0.05) # X ticks 0.0 to 0.2 step 0.05
    y_ticks = np.arange(0.0, 2.5 + 0.5, 0.5)  # Y ticks 0.0 to 2.5 step 0.5
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set axis limits with a small margin
    x_margin = (x_ticks.max() - x_ticks.min()) * 0.05 if len(x_ticks)>1 else 0.01
    y_margin = (y_ticks.max() - y_ticks.min()) * 0.05 if len(y_ticks)>1 else 0.1
    ax.set_xlim(x_ticks.min() - x_margin, x_ticks.max() + x_margin)
    ax.set_ylim(y_ticks.min() - y_margin, y_ticks.max() + y_margin)

    # --- Legend and Grid ---
    if not df_model1.empty or not df_model2.empty: # Only show legend if there's data
        ax.legend(loc='upper right') # Legend location
    ax.grid(True) # Enable grid
    plt.tight_layout() # Adjust layout

    # --- Save Plot ---
    try:
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Plot saved to file: {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}", file=sys.stderr)

    # Optional: show plot
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates average payoff vs noise plot from 'results.csv'.")
    # Input file is hardcoded now
    # parser.add_argument("csv_file", help="Path to the input CSV file with simulation results.")
    parser.add_argument("-o", "--output", default="payoff_vs_noise_plot.png",
                        help="Output filename for the plot (default: payoff_vs_noise_plot.png)")

    args = parser.parse_args()

    # Hardcode the input filename
    input_csv_file = "results.csv"

    # Check if input file exists
    if not os.path.isfile(input_csv_file):
        print(f"Error: Input file '{input_csv_file}' not found.", file=sys.stderr)
        print("Ensure the file is in the same directory as the script.", file=sys.stderr)
        sys.exit(1)

    # Call the plot generation function
    generate_plot(input_csv_file, args.output)
