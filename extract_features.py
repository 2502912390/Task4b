import os
import utils
import librosa
import pandas as pd
from sklearn import preprocessing
import config
from tqdm import tqdm
from LASS_codes.models.clap_encoder import CLAP_Encoder
import torchaudio
import soundfile as sf
from utils import (
    load_ss_model,
    parse_yaml
)
import torch
import numpy as np
from typing import Dict

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
def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    spec = np.abs(librosa.stft(_y, n_fft=_nfft, hop_length=_hop)) ** 1  # `power=1` 计算幅度谱
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, fmin=_fmin, fmax=_fmax)
    mel_spec = np.dot(mel_basis, spec)
    
    return mel_spec


#保存一整段音频的mel和soft_label 共98*2个 保存到features_mbe
def extract_data(dev_file, audio_path, annotation_path, feat_folder):#dev_file=development_split.csv 包含所有音频文件
# Extract features for all audio files
    # User set parameters
    is_mono = True

    if config.lass_sr == 32000:
        config_yaml = config.separate_audio_config_yaml_32k
        checkpoint_path = config.separate_audio_checkpoint_path_32k
    elif config.lass_sr == 16000:
        config_yaml = config.separate_audio_config_yaml_16k
        checkpoint_path = config.separate_audio_checkpoint_path_16k
    configs = parse_yaml(config_yaml)
    
    # Load model
    query_encoder = CLAP_Encoder().eval()
    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(config.device)
    pl_model.eval()

    with torch.no_grad():
        files = pd.read_csv(dev_file)['filename']
        for file in files:
            print(file)
            audio_name = file.split(os.path.sep)[-1]

            # MEL features
            audio, sr = utils.load_audio(os.path.join(audio_path, file+'.wav'), mono=is_mono, fs=config.sample_rate) #加载音频 一整段 这里也是直接加载44100采样率的
            audio = torch.tensor(audio).unsqueeze(0)
            # print(sr) # 44100
            # print(audio.shape) #torch.Size([1,13283328])
            
            # audio和训练好的lass sr不一样 所以要重采样统一
            if sr != config.lass_sr:
                audio_2 = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=config.lass_sr)
            else:
                audio_2 = audio
            audio_2 = audio_2.float()
            # print(audio_2.shape) #([1,9638697])

            class_segment = []
            for caption in config.labels_soft:# 这里改成labels_soft 17个类别 为了匹配上标签
                conditions = pl_model.query_encoder.get_query_embed( 
                                    modality='text',
                                    text=[caption],
                                    device=config.device 
                                )
                n_sample = config.lass_sr * config.lass_duration #训练好了的lass处理的dur是固定的 所以要对audio进行切分处理
                nums = audio_2.shape[-1] // n_sample #一个音频片段需要分几次来进行lass
                # print(nums) # 30  数量不固定

                final_segment = []
                for i in range(nums):
                    segment = audio_2[:, i*n_sample:(i+1)*n_sample] #对划分的音频片段再划分为10s段
                    segment = segment.to(config.device)
                    # print(segment.shape)#([1, 320000])
                    input_dict = {
                                    "mixture": segment[None, :, :],
                                    "condition": conditions,
                                }
                    
                    outputs = pl_model.ss_model(input_dict)
                    sep_segment = outputs["waveform"]
                    sep_segment = sep_segment.squeeze(0)
                    # print(sep_segment.shape)#([1, 320000])

                    # concatenate
                    final_segment.append(sep_segment.cpu()) #将一个音频片段的所有划分（10s）分离后的结果cat起来 这里每一段之间没有overlap
                    del segment, input_dict, outputs, sep_segment
                    torch.cuda.empty_cache()
                  
                if (audio_2.shape[-1] - (i+1)*n_sample) > 0:
                    segment = audio_2[:, (i+1)*n_sample: ]
                    segment = segment.to(config.device)
                    rest_sample = segment.shape[-1]

                    segment_pad = torch.zeros((1, config.lass_sr * config.lass_duration)).to(config.device)
                    segment_pad[:, :rest_sample] = segment
                    input_dict = {
                                    "mixture": segment_pad[None, :, :],
                                    "condition": conditions,
                                }
                    
                    outputs = pl_model.ss_model(input_dict)
                    sep_segment = outputs["waveform"]
                    sep_segment = sep_segment.squeeze(0)
                    sep_segment = sep_segment[:, :rest_sample]
                    final_segment.append(sep_segment.cpu())
                    del segment, input_dict, outputs, sep_segment
                    torch.cuda.empty_cache()

                final_segment = torch.cat(final_segment, dim=-1)
                
                # lass处理完了之后采样率又变回44100
                if sr != config.lass_sr:
                    final_segment = torchaudio.functional.resample(final_segment, orig_freq=config.lass_sr, new_freq=sr)

                final_segment = final_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                # print(final_segment.shape) #(13283330,) 这个也不固定

                mel = extract_mbe(final_segment, config.sample_rate, config.nfft, config.hop_size, config.nb_mel_bands, config.fmin, config.fmax).T # [nmel, nframes]
                # print(mel.shape)#(1507,64) 这个也不固定
                class_segment.append(mel)

            class_segment = np.stack(class_segment, axis=0)
            # print(class_segment.shape) #(17, 1507, 64)
            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            np.savez(tmp_feat_file, class_segment) #保存mel到feat_folder为npz格式

            # Extraction SOFT Annotation
            nframes = class_segment.shape[1]
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

            if X_train is None:
                X_train = tmp_mbe
            else:
                X_train = np.concatenate((X_train, tmp_mbe), 0) #这样拼接是否正确？

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
        np.savez(normalized_feat_file, X_train, X_val)# 一折的训练+验证数据保存到development/features

        # For the test data save individually
        for file in test_files:
            audio_name = file.split('/')[-1]

            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            dmp = np.load(tmp_feat_file)
            tmp_mbe = dmp['arr_0']
            X_test = scaler.transform(tmp_mbe)
            normalized_test_file = os.path.join(output_folder, 'test_{}_fold{}.npz'.format(audio_name, fold))
            np.savez(normalized_test_file, X_test)# 保存测试数据
        
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
    audio_path = '/root/autodl-fs/dataset/MAESTRO_Real/development_audio'
    annotation_path = '/root/autodl-fs/dataset/MAESTRO_Real/development_annotation'
    dev_file = 'development_split.csv'
    
    # Output
    feat_folder = '/root/autodl-fs/dataset/MAESTRO_Real/features_mbe_lass/'
    # utils.create_folder(feat_folder)

    # # Extract mel features for all the development data
    # extract_data(dev_file, audio_path, annotation_path, feat_folder)#对整段音频文件保存mel和其label的np格式到feat_folder
    # os.system("/usr/bin/shutdown")

    # Normalize data into folds
    output_folder = '/root/autodl-fs/dataset/MAESTRO_Real/development/lass_concat_features'
    utils.create_folder(output_folder)
    fold_normalization(feat_folder, output_folder)# 对数据分折 一折内的训练+验证 测试保存为一个文件 并保存到development/features
    
    # Merge Soft Labels annotations
    # output_folder = '/root/autodl-fs/dataset/MAESTRO_Real/development/soft_labels'
    # utils.create_folder(output_folder)
    # merge_annotations_into_folds(feat_folder, 'soft', output_folder)# 对标签分折 并保存到development/soft_labels
    

