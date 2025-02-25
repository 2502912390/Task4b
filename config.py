import os
device = 'cuda'
posterior_thresh = 0.5

lass_sr = 32000 #lass处理的采样率
lass_duration = 10 #lass处理的长度

sample_rate = 44100 # 采样率是44100
hop_size = 8820 
nfft = int(hop_size * 2)
nb_mel_bands = 64
fmin = 50 
fmax = 14000 

segment = int(39.9 * sample_rate) 

# 17 classes
labels_soft = ['birds_singing', 'car', 'people talking', 'footsteps', 'children voices', 'wind_blowing',
          'brakes_squeaking', 'large_vehicle', 'cutlery and dishes', 'furniture dragging', 'coffee machine',
          'metro approaching', 'metro leaving', 'door opens/closes', 'announcement', 'shopping cart',
          'cash register beeping']
class_labels_soft = {
    'birds_singing': 0,
    'car': 1,
    'people talking': 2,
    'footsteps': 3,
    'children voices': 4,
    'wind_blowing': 5,
    'brakes_squeaking': 6,
    'large_vehicle': 7,
    'cutlery and dishes': 8,
    'furniture dragging': 9,
    'coffee machine': 10,
    'metro approaching': 11,
    'metro leaving': 12,
    'door opens/closes': 13,
    'announcement': 14,
    'shopping cart': 15,
    'cash register beeping': 16
}
classes_num_soft = len(labels_soft) #17



# For the hard labels we have 11 classes 只在测试的时候用到？？？
labels_hard = ['birds_singing', 'car', 'people talking', 'footsteps', 'children voices', 'wind_blowing',
          'brakes_squeaking', 'large_vehicle', 'cutlery and dishes', 'metro approaching', 'metro leaving']
class_labels_hard = {
    'birds_singing': 0,
    'car': 1,
    'people talking': 2,
    'footsteps': 3,
    'children voices': 4,
    'wind_blowing': 5,
    'brakes_squeaking': 6,
    'large_vehicle': 7,
    'cutlery and dishes': 8,
    'metro approaching': 9,
    'metro leaving': 10,
    }
classes_num_hard = len(labels_hard)

#预训练模型的路径
music_speech_path = '/root/autodl-fs/pretrained_models/LASS/music/music_speech_audioset_epoch_15_esc_89.98.pt'
vocab_file = '/root/autodl-fs/pretrained_models/LASS/roberta-base/model/vocab.json'
merges_file = '/root/autodl-fs/pretrained_models/LASS/roberta-base/model/merges.txt'
RobertaModel_path = '/root/autodl-fs/pretrained_models/LASS/roberta-base/model/roberta-base'

separate_audio_config_yaml_16k = '/root/autodl-fs/pretrained_models/LASS/resunet/Fsd_Clo_Caps_Autotest_ResUNet_16k.yaml'
separate_audio_checkpoint_path_16k = '/root/autodl-fs/pretrained_models/LASS/resunet/resunet_with_dprnn_16k/resunet_with_dprnn_16k/model-epoch=19-val_sdr=8.1018.ckpt'
separate_audio_config_yaml_32k = '/root/autodl-fs/pretrained_models/LASS/resunet/Fsd_Clo_Caps_Autotest_ResUNet_32k.yaml'
separate_audio_checkpoint_path_32k = '/root/autodl-fs/pretrained_models/LASS/resunet/resunet_with_dprnn_32k/resunet_with_dprnn_32k/model-epoch=01-val_sdr=8.6049.ckpt'

# ATST
pretrained_path = '/root/autodl-fs/pretrained_models/pretrained_models/stage_2_wo_external.ckpt'
model_config_path = '/root/autodl-fs/pretrained_models/pretrained_models/stage2.yaml'

# 用于测试的相关文件
ground_truth='/autodl-fs/data/Task4b/metadata/gt_dev.csv'
audio_durations= '/autodl-fs/data/Task4b/metadata/development_metadata.csv'

# 用于训练的数据和标签(lass)
lass_development_feature = '/root/autodl-fs/dataset/MAESTRO_Real/development/lass_concat_features'
lass_development_soft_labels = '/root/autodl-fs/dataset/MAESTRO_Real/development/lass_soft_labels'
# 普通
development_feature = '/root/autodl-fs/dataset/MAESTRO_Real/development/features/'
development_soft_labels = '/root/autodl-fs/dataset/MAESTRO_Real/development/soft_labels/'
# atst emb
atst_emb = '/root/autodl-fs/dataset/MAESTRO_Real/development/atst_emb_concat/'

# 结果保存
task = "tq_sed_17cls" 
result_path = '/root/autodl-fs/LOGANDRESULT/Task4b/result' 
output_model = os.path.join(result_path, task, 'model')
output_folder = os.path.join(result_path, task, 'dev_txt_scores')
output_folder_soft = os.path.join(result_path, task, 'dev_txt_scores_soft')
