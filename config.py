import os
import torch

device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
# device = 'cpu'
posterior_thresh = 0.5
sample_rate = 44100 # 采样率是44100
hop_size = 8820 

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

# For the hard labels we have 11 classes
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


# 用于测试的相关文件
ground_truth='/autodl-fs/data/Task4b/metadata/gt_dev.csv'
audio_durations= '/autodl-fs/data/Task4b/metadata/development_metadata.csv'

development_feature = '/root/autodl-fs/dataset/MAESTRO_Real/development/features/'
development_soft_labels = '/root/autodl-fs/dataset/MAESTRO_Real/development/soft_labels/'

# 结果保存
task = "0302_hard_label" 
result_path = '/root/autodl-fs/LOGANDRESULT/Task4b' 
output_model = os.path.join(result_path, task, 'model')
output_folder = os.path.join(result_path, task, 'dev_txt_scores')
output_folder_soft = os.path.join(result_path, task, 'dev_txt_scores_soft')
