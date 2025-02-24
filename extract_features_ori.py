import os
import numpy as np
import utils
import librosa
import pandas as pd
from sklearn import preprocessing
import config


# -----------------------------------------------------------------------
# Annotation extraction
# -----------------------------------------------------------------------
def load_labels(file_name, nframes):
    annotations = []
    for l in open(file_name):
        words = l.strip().split('\t')
        annotations.append([float(words[0]), float(words[1]), config.class_labels_soft[words[2]], float(words[3])])

    # Initialize label matrix
    label = np.zeros((nframes, len(config.class_labels_soft)))
    tmp_data = np.array(annotations)
    
    frame_start = np.floor(tmp_data[:, 0] * config.sample_rate / config.hop_size).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * config.sample_rate / config.hop_size).astype(int)
    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = tmp_data[:, 3][ind]

    return label

# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------
# def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
#     spec, _ = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_hop, power=1)
#     mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, fmin=_fmin, fmax=_fmax)

#     return np.dot(mel_basis, spec)


def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    # 计算 STFT 频谱
    spec = np.abs(librosa.stft(_y, n_fft=_nfft, hop_length=_hop)) ** 1  # `power=1` 计算幅度谱
    
    # 计算 Mel 滤波器
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, fmin=_fmin, fmax=_fmax)
    
    # 计算 Mel 频谱
    mel_spec = np.dot(mel_basis, spec)
    
    return mel_spec

#保存一整段音频的mel和soft_label 共98*2个 保存到features_mbe
def extract_data(dev_file, audio_path, annotation_path, feat_folder):#dev_file=development_split.csv 包含所有音频文件
# Extract features for all audio files
    # User set parameters
    hop_len = config.hop_size
    fs = config.sample_rate
    
    nfft = int(hop_len*2)
    nb_mel_bands = 64 
    is_mono = True
    fmin = 50
    fmax = 14000
    
    files = pd.read_csv(dev_file)['filename']
    for file in files:
        audio_name = file.split(os.path.sep)[-1]
        # MEL features
        y, sr = utils.load_audio(os.path.join(audio_path, file+'.wav'), mono=is_mono, fs=fs) #加载音频 一整段 这里也是直接加载44100采样率的
        mbe = extract_mbe(y, sr, nfft, hop_len, nb_mel_bands, fmin, fmax).T #计算mel频谱
        tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
        np.savez(tmp_feat_file, mbe)#保存mel到feat_folder为npz格式

        nframes = mbe.shape[0]
               
        # Extraction SOFT Annotation
        annotation_file_soft = os.path.join(annotation_path, 'soft_labels_' + file + '.txt')
        annotations_soft = load_labels(annotation_file_soft, nframes)
        tmp_lab_file = os.path.join(feat_folder, '{}_soft.npz'.format(audio_name))
        np.savez(tmp_lab_file, annotations_soft)# 保存对应的标签

# 对整段音频提取的
def fold_normalization(feat_folder, output_folder):
    for fold in np.arange(1, 6):
        name = str(fold)
        # Load data 这几个文件规定了每一折中的train val test文件
        train_files = pd.read_csv('development_folds/fold{}_train.csv'.format(name))['filename'].tolist()
        val_files = pd.read_csv('development_folds/fold{}_val.csv'.format(name))['filename'].tolist()
        test_files = pd.read_csv('development_folds/fold{}_test.csv'.format(name))['filename'].tolist()

        X_train, X_val = None, None
        for file in train_files:#每一折里面的训练集数据拼接
            audio_name = file.split('/')[-1]
            
            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            dmp = np.load(tmp_feat_file)
            tmp_mbe = dmp['arr_0']
            print(tmp_mbe.shape)

            if X_train is None:
                X_train = tmp_mbe
            else:
                X_train = np.concatenate((X_train, tmp_mbe), 0)
            print("*******")
            print(X_train.shape)
        print("Over")

        for file in val_files:#每一折里面的验证集数据拼接
            audio_name = file.split('/')[-1]

            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            dmp = np.load(tmp_feat_file)
            tmp_mbe = dmp['arr_0']
            if X_val is None:
                X_val = tmp_mbe
            else:
                X_val = np.concatenate((X_val, tmp_mbe), 0)

        # Normalize the training data, and scale the testing data using the training data weights
        scaler = preprocessing.StandardScaler()# 对数据进行标准化
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        normalized_feat_file = os.path.join(output_folder, 'merged_mbe_fold{}.npz'.format(fold))
        # np.savez(normalized_feat_file, X_train, X_val)# 一折的训练+验证数据保存到development/features

        # For the test data save individually
        for file in test_files:
            audio_name = file.split('/')[-1]

            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            dmp = np.load(tmp_feat_file)
            tmp_mbe = dmp['arr_0']
            X_test = scaler.transform(tmp_mbe)
            normalized_test_file = os.path.join(output_folder, 'test_{}_fold{}.npz'.format(audio_name, fold))
            # np.savez(normalized_test_file, X_test)# 保存测试数据
        
        print(f'\t{normalized_feat_file}')
        print(f'\ttrain {X_train.shape} val {X_val.shape}')



def merge_annotations_into_folds(feat_folder, labeltype, output_folder):
    for fold in np.arange(1, 6):
        name = str(fold)

        # Load data
        train_files = pd.read_csv('development_folds/fold{}_train.csv'.format(name))['filename'].tolist()
        val_files = pd.read_csv('development_folds/fold{}_val.csv'.format(name))['filename'].tolist()
        test_files = pd.read_csv('development_folds/fold{}_test.csv'.format(name))['filename'].tolist()

        Y_train,  Y_val = None, None
        for file in train_files:
            audio_name = file.split('/')[-1]

            tmp_lab_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_name, labeltype))
            dmp = np.load(tmp_lab_file)
            label_mat = dmp['arr_0']
            if Y_train is None:
                Y_train = label_mat
            else:
                Y_train = np.concatenate((Y_train, label_mat), 0)

        for file in val_files:
            audio_name = file.split('/')[-1]

            tmp_lab_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_name, labeltype))
            dmp = np.load(tmp_lab_file)
            label_mat = dmp['arr_0']
            if Y_val is None:
                Y_val = label_mat
            else:
                Y_val = np.concatenate((Y_val, label_mat), 0)

        lab_file = os.path.join(output_folder, 'merged_lab_{}_fold{}.npz'.format(labeltype, fold))
        np.savez(lab_file, Y_train, Y_val)
        
        for file in test_files:
            audio_name = file.split('/')[-1]

            tmp_lab_file = os.path.join(feat_folder,'{}_{}.npz'.format(audio_name, labeltype))
            dmp = np.load(tmp_lab_file)
            label_mat = dmp['arr_0']
            lab_file = os.path.join(output_folder, 'lab_{}_{}_fold{}.npz'.format(labeltype, audio_name, fold ))
            np.savez(lab_file, label_mat)

        print(f'\t{lab_file}')
        print(f'\ttrain {Y_train.shape} val {Y_val.shape} ')



# ########################################
#              Main script starts here
# ########################################

if __name__ == '__main__':
    # path to all the data
    # audio_path = '/root/autodl-fs/dataset/MAESTRO_Real/development_audio'
    # annotation_path = '/root/autodl-fs/dataset/MAESTRO_Real/development_annotation'
    # dev_file = 'development_split.csv'
    
    # # Output
    # feat_folder = 'features_mbe/'
    # utils.create_folder(feat_folder)

    # # Extract mel features for all the development data
    # extract_data(dev_file, audio_path, annotation_path, feat_folder)#对整段音频文件保存mel和其label的np格式到feat_folder

    # Normalize data into folds
    output_folder = 'development/features'
    utils.create_folder(output_folder)
    fold_normalization("/root/autodl-fs/dataset/MAESTRO_Real/features_mbe/", output_folder)# 对数据分折 一折内的训练+验证 测试保存为一个文件 并保存到development/features
    
    # Merge Soft Labels annotations
    # output_folder = 'development/soft_labels'
    # utils.create_folder(output_folder)
    # merge_annotations_into_folds(feat_folder, 'soft', output_folder)# 对标签分折 并保存到development/soft_labels
    

