
import numpy as np
import pandas as pd
import os

path = 'shemo'
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}

def get_emotion_label(file_name):
  
    emo_code = file_name[3] 
    return emo_codes[emo_code]

def opensmile_cmd():
    this_path_output = 'NewVersion/speech_featuresarrayFloat16.csv'
    features = []
    labels = []
    processed_files = []  

    for filename in sorted(os.listdir(path)):  
        label = get_emotion_label(filename)
        if label == 5:  
            continue

        opensmile_config_path = 'opensmile-3.0-win-x64/config/misc/emo_large.conf'
        single_feat_path = 'NewVersion/speech_featuresarrayFloat16.csv'
        cmd = f'SMILExtract -C {opensmile_config_path} -I {path}/{filename} -O {single_feat_path}'
        os.system(cmd)
        processed_files.append(filename)
        labels.append(label)

    # with open('NewVersion/processed_filesshemov6.txt', 'w') as f:
    #     for file in processed_files:
    #         f.write(file + '\n')

    df = pd.read_csv(this_path_output, skiprows=6559)
    features = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    if len(labels) > features.shape[0]:
        print(f"Warning: Removing last label ({labels[-1]}) to match feature count.")
        labels = labels[:-1]
    # np.save('NewVersion/new_opensmile_emolarge_featuresarray.npy', np.array(features, dtype=np.float32))
    # np.save('NewVersion/new_opensmile_labelsarray.npy', np.array(labels, dtype=np.int32))
    np.save('NewVersion/new_opensmile_emolarge_featuresarrayFloat16.npy', np.array(features, dtype=np.float16))
    np.save('NewVersion/new_opensmile_labelsarrayFloat16.npy', np.array(labels, dtype=np.int32))


    print(f"Feature shape: {np.array(features).shape}")
    print(f"Label shape: {len(labels)}")


if __name__ == '__main__':
    opensmile_cmd()


