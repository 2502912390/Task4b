import os
import numpy as np
import librosa
import wave
import config as config
from dcase_util.containers import metadata
import torch
import torch.nn as nn

from typing import Dict
import yaml
from LASS_codes.models.audiosep import AudioSep, get_model_class

eps = np.finfo(np.float64).eps


def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array
    Supports 24-bit wav-format

    Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python

    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None



def create_folder(_fold_path):
    if not os.path.exists(_fold_path):
        os.makedirs(_fold_path)


# 将长数据划分为subdivs大小
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


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def parse_yaml(config_yaml: str) -> Dict:

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)
    

def load_ss_model(
    configs: Dict,
    checkpoint_path: str,
    query_encoder: nn.Module
) -> nn.Module:
    r"""Load trained universal source separation model.

    Args:
        configs (Dict)
        checkpoint_path (str): path of the checkpoint to load
        device (str): e.g., "cpu" | "cuda"

    Returns:
        pl_model: pl.LightningModule
    """

    ss_model_type = configs["model"]["model_type"]
    input_channels = configs["model"]["input_channels"]
    output_channels = configs["model"]["output_channels"]
    condition_size = configs["model"]["condition_size"]
    dprnn = configs['model']['dprnn']
    dprnn_layers = configs['model']['dprnn_layers']
    dprnn_hidden = configs['model']['dprnn_hidden']

    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
        dprnn=dprnn,
        dprnn_layers=dprnn_layers,
        dprnn_hidden=dprnn_hidden
    )

    # Load PyTorch Lightning model
    pl_model = AudioSep.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        ss_model=ss_model,
        waveform_mixer=None,
        query_encoder=query_encoder,
        loss_function=None,
        optimizer_type=None,
        learning_rate=None,
        lr_lambda_func=None,
        map_location=torch.device('cpu'),
    )

    return pl_model