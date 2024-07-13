import pandas as pd


if __name__ == '__main__':
    dataset = 'MMWHS'
    comparison_csv = f'../documents/ablation/partition/ablation_partition.csv'
    compare_latex_csv = f'../documents/ablation/partition/ablation_partition_latex.csv'
    # read in the csv file of the comparison results
    df = pd.read_csv(comparison_csv, index_col=0)
    index = df.index
    num_col = df.shape[1]
    # cols = ['MYO', 'LV', 'RV', 'AVG', 'MYO', 'LV', 'RV', 'AVG', 'MYO', 'LV', 'RV', 'AVG']
    cols = ['DC', 'HD95']
    # generate the dataframe for the new data
    df_new = pd.DataFrame(columns=cols, index=index)
    for idx in index:
        row = df.loc[idx]
        for i in range(num_col // 2):
            df_new.loc[idx][i] = f"{row[2 * i]}$/pm${row[2 * i + 1]}"
    # meta comment: remember to replace all the / with \ so that \pm works :)
    df_new.to_csv(compare_latex_csv)
