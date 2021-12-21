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


def show_dl(dl, attr=("shape"), first_n=1):
    counter = 0
    for x in dl:
        print("-"*20)
        if "shape" in attr:
            print({k: v.shape for k, v in x.items()})
        if "all" in attr:
            print(f"x: {x}")

        counter += 1
        if counter == first_n:
            break
