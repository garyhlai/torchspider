from matplotlib import pyplot as plt
import os


def show_data_img(data):
    plt.figure()
    plt.imshow(data['image'])
    plt.figtext(0, 0, f"score: {data['target']}", fontsize=10)


def find_incremental_filename(path):
    i = 0
    while os.path.exists(f"{path}/learner{i}.pkl"):
        i += 1
    return f"{path}/learner{i}.pkl"


def shuffle_df(df):
    """
    shuffle a pandas dataframe
    """
    return df.sample(frac=1).reset_index(drop=True)
