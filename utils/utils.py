import os
import numpy as np
import pandas as pd
import os
import shutil


def map_lb():
    x = 30.18
    y = -97.92
    z = 0
    for path, _, files in os.walk('rawdata'):
        for file in files:
            if file[-4:] == '.csv':
                df = pd.read_csv('rawdata/' + file)
                if np.min(df['latitude']) < x:
                    x = np.min(df['latitude'])
                if np.min(df['longitude']) < y:
                    y = np.min(df['longitude'])
                if np.min(df['altitude']) < z:
                    z = np.min(df['altitude'])
                print(np.min(df['latitude']), np.min(df['longitude']), np.min(df['altitude']))
    print(x, y, z)


def save_mat():
    input_path = 'data/March/'
    for path, _, files in os.walk(input_path):
        for file in files:
            if file[-4:] == '.csv' and file != 'stat.csv' and file != 'coordinate.csv' and file != 'time.csv':
                print(file)
                # if os.path.exists(input_path + 'mat' + file[0:2] + '.npy'):
                #     continue
                df = pd.read_csv(input_path + file)
                max_bx = np.max(df['bx']) + 1
                max_by = np.max(df['by']) + 1
                mat = np.zeros((max_bx, max_by), dtype=int)
                print(np.shape(mat), np.sum(mat))
                for index, row in df.iterrows():
                    mat[row['bx'], row['by']] = mat[row['bx'], row['by']] + 1
                print(index, np.sum(mat))
                # np.save(input_path + 'mat' + file[0:2] + '.npy', mat)
                # print(mat)


def pre_process():
    DRIVE_DATA_DIR = "../dataset/"
    fileDir = "RollSparse"
    filePostfix = "RS"
    tempFeatUser = np.loadtxt(os.path.join(DRIVE_DATA_DIR, f"{fileDir}/feat{filePostfix}.csv"), delimiter=',')
    tempFeatPing = np.loadtxt(os.path.join(DRIVE_DATA_DIR, f"{fileDir}/feat{filePostfix}Ping.csv"), delimiter=',')
    tempFeatUser = tempFeatUser[:, 6:-18].transpose() + 1
    tempFeatPing = tempFeatPing[:, 6:-18].transpose() + 1
    os.makedirs(f"../dataset/temp/{fileDir}", exist_ok=True)
    np.savetxt(f"../dataset/temp/{fileDir}/feat{filePostfix}.csv", tempFeatUser, delimiter=',')
    np.savetxt(f"../dataset/temp/{fileDir}/feat{filePostfix}Ping.csv", tempFeatPing, delimiter=',')
    shutil.copy2(os.path.join(DRIVE_DATA_DIR, f"{fileDir}/adj{filePostfix}.csv"), f"../dataset/temp/{fileDir}/adj{filePostfix}.csv")


if __name__ == '__main__':
    pre_process()
    # feat = np.loadtxt('featPing.csv', delimiter=',')
    # np.savetxt('featping.csv', feat.T, delimiter=',', fmt='%d')
    # print(feat.shape)
    # input_path = 'data/March/'
    # for path, _, files in os.walk(input_path):
    #     for file in files:
    #         if file[-4:] == '.csv' and file != 'stat.csv' and file != 'coordinate.csv' and file != 'time.csv':
    #             print(file)
    # input_path = 'data/March/'
    # name = [str(i).zfill(2) for i in range(1, 31)]
    # for i in name:
    #     df = pd.read_csv(input_path + i + '.csv')
    #     print(i, df.shape[0])