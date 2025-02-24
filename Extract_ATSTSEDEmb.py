

import torchaudio
import torch
import os
import yaml
import config
import librosa
import pandas as pd
import numpy as np
import torch.nn as nn
import utils
from tqdm import tqdm
from torchaudio.transforms import AmplitudeToDB
from desed_task.dataio.datasets_atst_sed import SEDTransform, ATSTTransform, read_audio
from desed_task.utils.scaler import TorchScaler
from desed_task.nnet.CRNN_e2e import CRNN
import warnings
warnings.filterwarnings("ignore")

def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    spec = np.abs(librosa.stft(_y, n_fft=_nfft, hop_length=_hop)) ** 1  
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, fmin=_fmin, fmax=_fmax)
    mel_spec = np.dot(mel_basis, spec)
    
    return mel_spec

class ATSTSEDFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sed_feat_extractor = SEDTransform(config["feats"])
        self.scaler = TorchScaler(
            "instance",
            config["scaler"]["normtype"],
            config["scaler"]["dims"],
        )
        self.atst_feat_extractor = ATSTTransform()

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def forward(self, mixture):
        mixture = mixture.unsqueeze(0)  # fake batch size
        sed_feats = self.sed_feat_extractor(mixture)  # (1 128 626) sed特征提取
        sed_feats = self.take_log(sed_feats)
        sed_feats = self.scaler(sed_feats)
        atst_feats = self.atst_feat_extractor(mixture)  # (1 64 1001) atst特征提取

        return sed_feats, atst_feats

class ATSTSEDInferencer(nn.Module):

    """Inference module for ATST-SED
    """
    def __init__(
            self,
            pretrained_path,
            model_config_path="./confs/stage2.yaml",
            overlap_dur=3, #重叠部分的时长
            hard_threshold=0.5):
        super().__init__()

        # Load model configurations
        with open(model_config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.config = config
        # Initialize model
        self.model = self.load_from_pretrained(pretrained_path, config)
        # Initialize feature extractor
        self.feature_extractor = ATSTSEDFeatureExtractor(config)

        # Initial parameters
        self.audio_dur = 10  # this value is fixed because ATST-SED is trained on 10-second audio, if you want to change it, you need to retrain the model
        self.overlap_dur = overlap_dur
        self.fs = config["data"]["fs"]

        # Unfolder for splitting audio into chunks
        self.unfolder = nn.Unfold(kernel_size=(self.fs * self.audio_dur, 1), stride=(self.fs * self.overlap_dur, 1))


    def load_from_pretrained(self, pretrained_path: str, config: dict):
        # Initializign model
        model = CRNN(
            unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"],
            **config["net"],
            model_init=config["ultra"]["model_init"],
            atst_dropout=config["ultra"]["atst_dropout"],
            atst_init=config["ultra"]["atst_init"],
            mode="teacher")
        # Load pretrained ckpt
        state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]
        ### get teacher model
        state_dict = {k.replace("sed_teacher.", ""): v for k, v in state_dict.items() if "teacher" in k}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def get_logmel(self, audio):

        sed_feats, atst_feats = self.feature_extractor(audio)
        return sed_feats[0].detach().cpu().numpy(), atst_feats[0].detach().cpu().numpy()

    def forward(self, mixture):

        # ATST-SED默认处理10s音频
        if (mixture.numel() // self.fs) <= self.audio_dur:
            inference_chunks = [mixture]
            padding_frames = 0
            mixture_pad = mixture.clone()
        else:
            # pad the mixtures
            mixture = mixture.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            # 以10s为单位长度 重叠3s 所能分离出来的片段 这里能不能不进重叠？
            total_chunks = (mixture.numel() - ((self.audio_dur - self.overlap_dur) * self.fs)) // (self.overlap_dur * self.fs) + 1
            # 填充后的音频总长度
            total_length = total_chunks * self.overlap_dur * self.fs + (self.audio_dur - self.overlap_dur) * self.fs
            mixture_pad = torch.nn.functional.pad(mixture, (0, 0, 0, total_length - mixture.numel()))
            # padding_frames = self.time2frame(total_length - mixture.numel())
            inference_chunks = self.unfolder(mixture_pad)
            inference_chunks = inference_chunks.squeeze(0).T #(11 160000) 分离出11块 每块10s 每秒16000 这一步就保证了输入到CRNN中的数据是固定的了

        # inference result for each chunk
        sed_results = []
        for chunk in inference_chunks:  # (160000)
            sed_feats, atst_feats = self.feature_extractor(chunk)  # (1 128 626) (1 64 1001)
            # sed_feats = sed_feats.transpose(1, 2)
            chunk_result = self.model(sed_feats, atst_feats).squeeze(0) #(1 156 128) 代表每10s的特征 第二个是事序维度
            sed_results.append(chunk_result)

        sed_results = torch.stack(sed_results) #堆叠起来 表示一个39.9s的音频被ATST-SED分离出来的特征
        return sed_results #(11 156 128)

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs)]
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data

def extract_atst_emb(dev_file, audio_path, save_folder):#dev_file=development_split.csv 包含所有音频文件
    inference_model = ATSTSEDInferencer(
                    config.pretrained_path,
                    config.model_config_path,
                    overlap_dur=3)
    
    files = pd.read_csv(dev_file)['filename']
    os.makedirs(save_folder, exist_ok=True)

    for file in files:
        print(file)
        audio_name = file.split(os.path.sep)[-1]
        # MEL features
        y, sr = utils.load_audio(os.path.join(audio_path, file+'.wav'), mono=True, fs=config.sample_rate)
        # print(y.shape) #(13283328,)
        audio_sep = split_in_seqs(y,config.segment) #正确的
        # print(audio_sep.shape) #(7, 1759590, 1)

        audio_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        audio_resampled_tensor = torch.from_numpy(audio_resampled).float()

        # 一段一段来处理
        for i in audio_sep.shape[0]:
            feature = inference_model(audio_resampled_tensor) #(clip_num 156 128)
            feature = feature.detach().cpu().numpy()
        
        # tmp_feat_file = os.path.join(save_folder, '{}.npz'.format(audio_name))
        # np.savez(tmp_feat_file, feature)

# interface
if __name__ == "__main__":
    audio_path = '/root/autodl-fs/dataset/MAESTRO_Real/development_audio'
    dev_file = 'development_split.csv'
    save_path = '/root/autodl-fs/dataset/MAESTRO_Real/atst_emb/'
    extract_atst_emb(dev_file,audio_path,save_path)