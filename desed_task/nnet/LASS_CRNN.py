from recipes.dcase2023_task4_baseline import config
from recipes.dcase2023_task4_baseline.LASS_codes.models.clap_encoder import CLAP_Encoder
import torch
import torchaudio.transforms
import numpy as np
import librosa
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from recipes.dcase2023_task4_baseline.LASS_codes.utils import (
    load_ss_model,
    parse_yaml,
)
from recipes.dcase2023_task4_baseline.models.tq_sed import CRNN_LASS_A
import torch.nn as nn


def take_log(mels):
    amp_to_db = AmplitudeToDB(stype="amplitude")
    amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
    return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

class LASS_CRNN(nn.Module):
    def __init__(
        self
    ):
        super(LASS_CRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.mel_spec = MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            win_length=2048,
            hop_length=256,
            f_min=0,
            f_max=8000,
            n_mels=128,
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        ).to(self.device)

        #这个的顺序要和label保持一致？？
        # self.labels = ['Vacuum cleaner','Frying','Blender',
        #               'Electric shaver/toothbrush','Running water',
        #               'Dog', 'Cat', 'Speech',
        #               'Alarm/bell/ringing','Dishes',
        #     ]

        self.labels =[  "Alarm_bell_ringing",
                        "Blender",
                        "Cat",
                        "Dishes",
                        "Dog",
                        "Electric_shaver_toothbrush",
                        "Frying",
                        "Running_water",
                        "Speech",
                        "Vacuum_cleaner"  ]

        self.lass_sr = config.lass_sr  # 16000
        self.lass_dur = config.lass_duration  # 10

        if self.lass_sr == 32000:
            config_yaml = config.config_yaml_32k
            checkpoint_path = config.checkpoint_path_32k
        elif self.lass_sr == 16000:
            config_yaml = config.config_yaml_16k
            checkpoint_path = config.checkpoint_path_16k

        self.configs_LASS = parse_yaml(config_yaml)

        #这部分冻结
        query_encoder = CLAP_Encoder().eval()
        self.pl_model = load_ss_model(
            configs=self.configs_LASS,
            checkpoint_path=checkpoint_path,
            query_encoder=query_encoder
        ).to(self.device).eval()

        self.model_sed = CRNN_LASS_A(classes_num=config.class_nums, cnn_filters=config.cnn_filters, rnn_hid=config.rnn_hid,
                            _dropout_rate=config._dropout_rate).to(self.device)

        self.fs=16000

    def forward(self, x):
        sep_mels = []
        for caption in self.labels:  # 这里的labels是否可以根据其对应的label来？ 其余的补0
            conditions = self.pl_model.query_encoder.get_query_embed(
                modality='text',
                text=[caption],
                device=self.device
            )

            n_sample = self.lass_sr * self.lass_dur  # 每个片段的长度
            nums = x.shape[-1] // n_sample  # 切分的片段数量

            for i in range(nums):
                segment = x[:, i * n_sample:(i + 1) * n_sample]  # 一个音频片段
                segment = segment.to(self.device)

                input_dict = {
                    "mixture": segment[:, None, :],  # 添加一个通道维度
                    "condition": conditions,
                }
                outputs = self.pl_model.ss_model(input_dict)
                sep_segment = outputs["waveform"]
                sep_segment = sep_segment.squeeze(0)

                if i == 0:
                    final_segment = sep_segment
                else:
                    final_segment = torch.cat((final_segment, sep_segment), dim=-1)

            # 处理剩余片段
            if (x.shape[-1] - (i + 1) * n_sample) > 0:
                segment = x[:, (i + 1) * n_sample:]
                segment = segment.to(self.device)  # print(segment.shape) [1, n]
                rest_sample = segment.shape[-1]

                segment_pad = torch.zeros((1, self.lass_sr * self.lass_dur)).to(self.device)
                segment_pad[:, :rest_sample] = segment
                input_dict = {
                    "mixture": segment_pad[None, :, :],
                    "condition": conditions,
                }

                outputs = self.pl_model.ss_model(input_dict)
                sep_segment = outputs["waveform"]
                sep_segment = sep_segment.squeeze(0)
                sep_segment = sep_segment[:, :rest_sample]
                final_segment = torch.cat((final_segment, sep_segment), dim=-1)

            if self.fs != self.lass_sr:
                final_segment = torchaudio.functional.resample(final_segment, orig_freq=self.lass_sr, new_freq=self.fs)

            final_segment = final_segment.squeeze(1)

            mel = self.mel_spec(final_segment)
            sep_mels.append(mel)  # 10 bs 128 626

        sep_mels = torch.stack(sep_mels, dim=1)  # (bs 10 128 626)
        sep_mels = take_log(sep_mels)

        #变成频谱图后还需要添加数据增强？？？

        strong, weak = self.model_sed(sep_mels)
        return strong,weak

if __name__ == '__main__':
    audio = torch.rand((1, 160000))
    model=LASS_CRNN()
    strong,weak = model(audio)
    print(strong.shape)
    print(weak.shape)
