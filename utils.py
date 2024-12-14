import numpy as np
import calendar
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import ast


def create_monthly_vector(year):
    """
    Creates a NumPy vector with 12 values, where each value is 1/x,
    and x is the number of days in the corresponding month of the given year.

    Args:
        year (int): The year for which to calculate the vector.

    Returns:
        numpy.ndarray: A NumPy array containing the 12 values.
    """

    days_in_month = [calendar.monthrange(year, month)[1] for month in range(1, 13)]
    return 1 / np.array(days_in_month)


def monthly_scale_vector(year):
    vector = create_monthly_vector(year)
    mapping_vector = {month: scale for month, scale in enumerate(vector, start=1)}
    return mapping_vector



def get_params():
    sp = pd.read_csv('sp2.csv')
    sp.set_index('month', inplace=True)
    sp = sp.map(ast.literal_eval)

    cp3 = pd.read_csv('cp3.csv')
    cp3.set_index('month', inplace=True)
    cp3['tot_range'] = cp3['tot_range'].apply(ast.literal_eval)
    cp3['norm_range'] = cp3['norm_range'].apply(ast.literal_eval)

    return sp, cp3


class SimOutput:
    def __init__(self, props, deltas, total_deficits, dates):
        self.props = props
        self.deltas = np.array(deltas)
        self.total_deficits = total_deficits
        self.dates = dates

        self.theta, self.var_theta, self.mu_def, self.var_def = self._basic_output()

        print(f"theta ~= {self.theta}")
        print(f"var_theta ~= {self.var_theta}")
        print(f"mu_def ~= {self.mu_def}")
        print(f"var_def ~= {self.var_def}")

        self.neg_delta_array = np.where(self.deltas < 0, self.deltas, 0)

    def _basic_output(self):
        theta = np.mean(self.props)
        var_theta = np.var(self.props)

        mu_def = np.mean(self.total_deficits)
        var_def = np.var(self.total_deficits)

        return theta, var_theta, mu_def, var_def

    def plot_avg_daily_delta(self):
        self.avg_delta = np.mean(self.deltas, axis=0)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(self.dates, self.avg_delta, label='Average Delta')

        plt.title("Average Simulated Electricity Generation - Demand, 2025")
        plt.xlabel("Date")
        plt.ylabel("kWh")
        ax.axhline(0, linestyle='--', c='green', label='0')

        myFmt = mdates.DateFormatter('%m/%Y')
        ax.xaxis.set_major_formatter(myFmt)

        # Define the center and radius of the circle
        center_date = mdates.date2num(self.dates[260])  # Example: Center at the 50th date
        center_value = 150  # Example: Center at the corresponding value
        radius_value = 100  # Example: Radius of 100 kWh

        # Convert radius to data coordinates (assuming y-axis is linear)
        circle = mpatches.Circle((center_date, center_value), radius_value, fill=False, edgecolor='red', lw=2,
                                 label='Average Delta drops here')

        # Add the circle to the axes
        ax.add_patch(circle)

        plt.legend(loc='lower left')
        plt.show()

    def plot_daily_neg_delta(self):
        """
        This tells us what the average ruin day deficit is for every day. That is, given that there is a ruin day on a particular day, what is the expected deficit.
        :return:
        """
        total_neg_daily_delta = np.sum(self.neg_delta_array, axis=0)
        fr_counts = np.sum(self.neg_delta_array < 0, axis=0)

        # Replace zeros in fr_counts with a small value to avoid division by zero
        fr_counts_safe = np.where(fr_counts == 0, 1, fr_counts)

        self.avg_neg_daily_delta = np.where(total_neg_daily_delta < 0, total_neg_daily_delta / fr_counts_safe, 0)

        plt.figure(figsize=(10, 8))
        plt.plot(self.dates, -self.avg_neg_daily_delta)
        plt.title("Average Daily Deficit")
        plt.xlabel("Date")
        plt.ylabel("Average Deficit")
        plt.show()

    def plot_monthly_neg_delta(self):
        """
        This gives us the distribution of and average of deficits occuring in each month
        """

        self.avg_neg_monthly_delta = pd.DataFrame()
        self.avg_neg_monthly_delta['date'] = self.dates
        self.avg_neg_monthly_delta['delta'] = self.avg_neg_daily_delta
        self.avg_neg_monthly_delta.set_index('date', inplace=True)

        self.monthly_avg_deficit = []

        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 10))
        for i, ax in enumerate(axs.flat):
            freq_data = self.avg_neg_monthly_delta.loc[self.avg_neg_monthly_delta.index.month == i + 1]
            mu = np.mean(freq_data)
            self.monthly_avg_deficit.append(mu)
            ax.hist(freq_data)
            # ax.axvline(mu, linestyle='--', c='red', label='Mean = {}'.format(mu))
            ax.set_title(f"Month {i + 1}")
            ax.set_xlabel("Deficit")
            ax.set_ylabel("Freq.")
            # plt.legend(loc='upper right')

        fig.suptitle("Monthly Average Deficit")
        fig.tight_layout()
        plt.show()

    def export(self, path):
        out_df = pd.DataFrame(columns=['props', 'deltas', 'deficits'])
        out_df['props'] = self.props
        out_df['deltas'] = self.deltas
        out_df['deficits'] = self.total_deficits
        out_df.to_csv(path)

    def output_report(self, name):
        with open('output_report.txt', 'w') as f:
            S = f"""NAME: {name}
=============
theta = {self.theta}
var_theta = {self.var_theta}
mu_def = {self.mu_def}
var_def = {self.var_def}
Monthly Average Deficit: {self.monthly_avg_deficit}"""
            f.write(S)



def plot_path(dates, gen, dem):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(dates, gen, c='b', label='Simulated Generation')
    ax.plot(dates, dem, c='r', label='Simulated Demand')
    plt.title("Simulated Electricity Generation v. Demand, 2025")
    plt.xlabel("Date")
    plt.ylabel("kWh")
    plt.legend(loc='upper left')

    myFmt = mdates.DateFormatter('%m/%Y')
    ax.xaxis.set_major_formatter(myFmt)
    plt.show()