# Funny libraries I use
import os 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
from scipy.stats import ttest_ind, f_oneway, ttest_rel
import pandas as pd
import seaborn as sea

# Part one, the great data nabbing. 

csv_directory = os.path.dirname(os.path.abspath(__file__))
csv_spot = os.path.join(csv_directory, "BIOE 340 - Final Project CSV.xlsx")
csv = pd.read_excel(csv_spot)
csv.to_csv('csv_file.csv', index=False)

# Part two, the funny function making

def funny_data_grapher(data):
    # grab columns
    available_vars = [col for col in data.columns if col != 'Date']
    
    print("Available variables to plot:")
    for i, var in enumerate(available_vars, 1):
        print(f"  {i}. {var}")
    
    # get input
    try:
        choice = int(input("\nEnter the number of the variable you'd like to plot: "))
        if choice < 1 or choice > len(available_vars):
            print(f"Error: Please enter a number between 1 and {len(available_vars)}")
            return
        variable = available_vars[choice - 1]
    except ValueError:
        print("Error: Please enter a valid number")
        return
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data[variable], marker='o')
    plt.xlabel('Date')
    plt.ylabel(variable)
    plt.title(f'{variable} vs Date')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def pre_vs_post_grapher(data):
    # define the comparison pairs
    pairs = [
        ('Heart Rate', 'Post Heart Rate'),
        ('Breath Rate', 'Post Breath Rate')
    ]
    
    print("Available comparisons:")
    for i, (pre, post) in enumerate(pairs, 1):
        print(f"  {i}. {pre} vs {post}")
    
    # get input
    try:
        choice = int(input("\nEnter the number of the comparison you'd like to plot: "))
        if choice < 1 or choice > len(pairs):
            print(f"Error: Please enter a number between 1 and {len(pairs)}")
            return
        pre_var, post_var = pairs[choice - 1]
    except ValueError:
        print("Error: Please enter a valid number")
        return
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data[pre_var], marker='o', label=pre_var)
    plt.plot(data['Date'], data[post_var], marker='s', label=post_var)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{pre_var} vs {post_var}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def data_statistics(data):
    # get all columns except Date
    numeric_vars = [col for col in data.columns if col != 'Date']
    
    # create dictionary to store stats
    stats_dict = {}
    
    for var in numeric_vars:
        stats_dict[var] = {
            'mean': data[var].mean(),
            'median': data[var].median(),
            'std': data[var].std()
        }
    
    # print results
    print("\nData Statistics:")
    print("-" * 80)
    for var, stats in stats_dict.items():
        print(f"{var}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")
        print()
    
    return stats_dict


def funny_visualizer(data):
    # define variables to test
    test_vars = [
        ('Amount of Creatine (grams)', 'Creatine'),
        ('Amount of Caffeine (mg)', 'Caffeine'),
        ('Nicotine Intake (Zyn: mg)', 'Nicotine'),
        ('Time spent in Sauna (minutes)', 'Sauna Time'),
        ('Exams/Papers (per week)', 'Exams/Papers')
    ]
    
    print("Variables to test against HR and BR:")
    for i, (col, label) in enumerate(test_vars, 1):
        print(f"  {i}. {label}")
    print(f"  {len(test_vars) + 1}. Meditation (Pre vs Post)")
    
    # get input
    try:
        choice = int(input("\nEnter the number of the variable to test: "))
        if choice < 1 or choice > len(test_vars) + 1:
            print(f"Error: Please enter a number between 1 and {len(test_vars) + 1}")
            return
    except ValueError:
        print("Error: Please enter a valid number")
        return
    
    if choice == len(test_vars) + 1:
        # Meditation case: 4 lines (HR, Post HR, BR, Post BR)
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Heart Rate'], marker='o', label='Heart Rate')
        plt.plot(data['Date'], data['Post Heart Rate'], marker='s', label='Post Heart Rate')
        plt.plot(data['Date'], data['Breath Rate'], marker='^', label='Breath Rate')
        plt.plot(data['Date'], data['Post Breath Rate'], marker='d', label='Post Breath Rate')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Meditation Effects on Heart Rate and Breath Rate')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        # Test variable against HR and BR: 3 lines (HR, BR, test variable)
        col, label = test_vars[choice - 1]
        
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Heart Rate'], marker='o', label='Heart Rate')
        plt.plot(data['Date'], data['Breath Rate'], marker='s', label='Breath Rate')
        plt.plot(data['Date'], data[col], marker='^', label=label)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Heart Rate, Breath Rate vs {label}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def comedic_color_splatter(data):
    # get all columns except Date
    vars_to_plot = [col for col in data.columns if col != 'Date']
    
    # create figure with larger size
    plt.figure(figsize=(14, 7))
    
    # plot each variable with a different color and marker
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h']
    for i, var in enumerate(vars_to_plot):
        marker = markers[i % len(markers)]
        plot_data = data[var]
        plot_label = var
        
        # Scale caffeine by dividing by 10
        if var == 'Amount of Caffeine (mg)':
            plot_data = data[var] / 10
            plot_label = 'Amount of Caffeine (per 10 mg)'
        
        plt.plot(data['Date'], plot_data, marker=marker, label=plot_label, linewidth=2, markersize=6)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comedic Color Splatter - All Variables', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def null_number_nlyzer(data):
    # define the 5 test variables (not meditation)
    test_vars = [
        ('Amount of Creatine (grams)', 'Creatine'),
        ('Amount of Caffeine (mg)', 'Caffeine'),
        ('Nicotine Intake (Zyn: mg)', 'Nicotine'),
        ('Time spent in Sauna (minutes)', 'Sauna Time'),
        ('Exams/Papers (per week)', 'Exams/Papers')
    ]
    
    print("\nNull Number Analyzer")
    print("=" * 80)
    
    for col, label in test_vars:
        print(f"\n{label}:")
        print("-" * 80)
        
        # get unique values for this variable
        unique_vals = sorted(data[col].unique())
        
        # for each unique value, calculate mean HR and BR
        for val in unique_vals:
            subset = data[data[col] == val]
            mean_hr = subset['Heart Rate'].mean()
            mean_br = subset['Breath Rate'].mean()
            count = len(subset)
            
            print(f"  {label} = {val}: HR mean = {mean_hr:.2f}, BR mean = {mean_br:.2f} (n={count})")
        
        print()


def statistical_significance_tester(data):
    # define the 5 test variables (not meditation)
    test_vars = [
        ('Amount of Creatine (grams)', 'Creatine'),
        ('Amount of Caffeine (mg)', 'Caffeine'),
        ('Nicotine Intake (Zyn: mg)', 'Nicotine'),
        ('Time spent in Sauna (minutes)', 'Sauna Time'),
        ('Exams/Papers (per week)', 'Exams/Papers')
    ]
    
    print("\nStatistical Significance Testing (vs Heart Rate)")
    print("=" * 80)
    print("Significance level: p < 0.05\n")
    
    for col, label in test_vars:
        print(f"{label}:")
        print("-" * 80)
        
        unique_vals = sorted(data[col].unique())
        
        # If binary (2 unique values), use t-test
        if len(unique_vals) == 2:
            group1 = data[data[col] == unique_vals[0]]['Heart Rate']
            group2 = data[data[col] == unique_vals[1]]['Heart Rate']
            
            t_stat, p_value = ttest_ind(group1, group2)
            significant = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
            
            print(f"  Test: Independent t-test")
            print(f"  {label} = {unique_vals[0]}: HR mean = {group1.mean():.2f}")
            print(f"  {label} = {unique_vals[1]}: HR mean = {group2.mean():.2f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Statistically Significant: {significant}")
        
        # If more than 2 values, use ANOVA
        else:
            groups = [data[data[col] == val]['Heart Rate'].values for val in unique_vals]
            f_stat, p_value = f_oneway(*groups)
            significant = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
            
            print(f"  Test: One-way ANOVA")
            for val in unique_vals:
                subset = data[data[col] == val]
                print(f"  {label} = {val}: HR mean = {subset['Heart Rate'].mean():.2f} (n={len(subset)})")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Statistically Significant: {significant}")
        
        print()
    
    # Test Meditation (paired t-test: Pre vs Post Heart Rate)
    print("Meditation (Heart Rate):")
    print("-" * 80)
    pre_hr = data['Heart Rate']
    post_hr = data['Post Heart Rate']
    
    t_stat, p_value = ttest_rel(pre_hr, post_hr)
    significant = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
    
    print(f"  Test: Paired t-test")
    print(f"  Pre-Meditation HR mean: {pre_hr.mean():.2f}")
    print(f"  Post-Meditation HR mean: {post_hr.mean():.2f}")
    print(f"  Difference: {(post_hr.mean() - pre_hr.mean()):.2f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Statistically Significant: {significant}")
    print()
    
    # Test Meditation (paired t-test: Pre vs Post Breath Rate)
    print("Meditation (Breath Rate):")
    print("-" * 80)
    pre_br = data['Breath Rate']
    post_br = data['Post Breath Rate']
    
    t_stat, p_value = ttest_rel(pre_br, post_br)
    significant = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
    
    print(f"  Test: Paired t-test")
    print(f"  Pre-Meditation BR mean: {pre_br.mean():.2f}")
    print(f"  Post-Meditation BR mean: {post_br.mean():.2f}")
    print(f"  Difference: {(post_br.mean() - pre_br.mean()):.2f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Statistically Significant: {significant}")
    print()


funny_data_grapher(csv)

pre_vs_post_grapher(csv)

data_statistics(csv)

funny_visualizer(csv)

comedic_color_splatter(csv)

null_number_nlyzer(csv)

statistical_significance_tester(csv)