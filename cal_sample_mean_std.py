import numpy as np
import pandas as pd
from glob import glob
import cv2


def save_mean_std(paths, df, modality, i):
    img_cat = []
    for path in paths:
        img_cat.append(cv2.imread(path))
    img_cat = np.array(img_cat)
    mean, std = img_cat.mean(), img_cat.std()
    df2 = pd.DataFrame([[modality.lower(), int(i), mean, std]],
                       columns=['modality', 'pat_id', 'mean', 'std'])
    df = df.append(df2, ignore_index=True)
    return df


def main():
    data = {'modality': [], 'pat_id': [], 'mean': [], 'std': []}
    df = pd.DataFrame(data)
    for modality in ['bSSFP', 't2']:
        for i in range(1, 6):
            paths = glob(f'D:\Work\ERC_project\Projects\data\mscmrseg\origin/testA/pat_{i}_{modality}_*.png')
            df = save_mean_std(paths, df, modality, i)
        for i in range(6, 46):
            paths = glob(f'D:\Work\ERC_project\Projects\data\mscmrseg\origin/trainA/pat_{i}_{modality}_*.png')
            df = save_mean_std(paths, df, modality, i)
    modality = 'lge'
    for i in range(1, 6):
        paths = glob(f'D:\Work\ERC_project\Projects\data\mscmrseg\origin/testB/pat_{i}_{modality}_*.png')
        df = save_mean_std(paths, df, modality, i)
    for i in range(6, 46):
        paths = glob(f'D:\Work\ERC_project\Projects\data\mscmrseg\origin/trainB/pat_{i}_{modality}_*.png')
        df = save_mean_std(paths, df, modality, i)
    df['pat_id'] = df['pat_id'].astype('int')
    df.to_csv('../../data/mscmrseg/mscmrseg_uint8_mean_std.csv', index=False)


if __name__ == '__main__':
    main()
