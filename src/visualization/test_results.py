import json

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv, find_dotenv

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
project_dir = Path(dotenv_path).parent

# load up the entries as environment variables
load_dotenv(dotenv_path)


if __name__ == '__main__':
    preds_fpaths = project_dir.glob('models/*_test_preds.json')

    # prepare data
    dfs = list()
    for preds_fpath in preds_fpaths:
        model = preds_fpath.name.split('_')[0]

        times_fpath = preds_fpath.parent/preds_fpath.name.replace('preds', 'times')

        with open(preds_fpath, 'r') as f:
            preds = json.load(f)

        with open(times_fpath, 'r') as f:
            times = json.load(f)

        sim = np.array([s.split('.')[0] for s in preds.keys()])
        y_hat = np.array(list(preds.values()))
        y_hat = pd.Series(y_hat, sim)

        sim = np.array([s.split('.')[0] for s in times.keys()])
        t = np.array(list(times.values()))
        t = pd.Series(t, sim)

        df = pd.DataFrame([y_hat, t]).T
        df = df.reset_index()
        df.columns = ['Sim', 'y_hat', 't']

        df['model'] = model

        dfs.append(df)

    df = pd.concat(dfs)

    df['y'] = df['Sim'].apply(lambda s: int(s.split('_')[-1]))
    df['error'] = (df['y'] - df['y_hat']).abs()

    # average performances
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)

    df.boxplot('error', 'model', ax=ax)

    ax.set_title('Performance on the test set')
    ax.set_xlabel('Models')
    ax.set_ylabel('Absolute error')

    # ax.set_ylim(-1, ax.get_ylim()[1])
    ax.set_yscale('log')
    ax.grid(False)

    fig.savefig(project_dir/'reports/figures/error_boxplot.png')
    plt.close(fig)

    # inference times
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)

    df.boxplot('t', 'model', ax=ax)

    ax.set_title('Prediction time')
    ax.set_xlabel('Models')
    ax.set_ylabel('Time')

    # ax.set_ylim(-1, ax.get_ylim()[1])
    ax.set_yscale('log')
    ax.grid(False)

    fig.savefig(project_dir/'reports/figures/time_boxplot.png')
    plt.close(fig)

    # performance by number of parts
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)

    for model in df['model'].unique():
        df_ = df[df['model'] == model]

        hits = (df_['y_hat'] + 0.5).astype(int) == df_['y']
        acc = hits.sum() / hits.size

        err_mean = df_.groupby('y')['error'].mean()
        err_std = df_.groupby('y')['error'].std()

        ax.fill_between(
            err_mean.index,
            err_mean + err_std,
            err_mean - err_std,
            alpha=0.5
        )
        ax.plot(err_mean, label=f"{model} (Acc. = {acc*100:.1f}%)")

    ax.set_xlim(0, 100)

    ax.set_title('Performance by number of parts in box')
    ax.set_xlabel('Number of parts')
    ax.set_ylabel('Absolute error')
    ax.set_yscale('log')

    ax.legend()
    ax.grid()

    fig.savefig(project_dir/'reports/figures/nparts_error_line.png')
    plt.close(fig)
