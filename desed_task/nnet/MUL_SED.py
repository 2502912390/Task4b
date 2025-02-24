from recipes.dcase2023_task4_baseline import config
import torch
from recipes.dcase2023_task4_baseline.models.tq_sed import CRNN_LASS_A
import torch.nn as nn

class MUL_SED(nn.Module):
    def __init__(
        self
    ):
        super(MUL_SED, self).__init__()
        self.device = config.device
        self.model_sed = CRNN_LASS_A(classes_num=config.class_nums, cnn_filters=config.cnn_filters, rnn_hid=config.rnn_hid,
                            _dropout_rate=config._dropout_rate).to(self.device)

    def forward(self, lass_feature):
        #变成频谱图后还需要添加数据增强？？？

        strong, weak = self.model_sed(lass_feature)
        return strong,weak

if __name__ == '__main__':
    audio = torch.rand((1,10,128,626)).to(config.device)
    model=MUL_SED().to(config.device)
    strong,weak = model(audio)
    print(strong.shape) #[bs 10 156]
    print(weak.shape) #[bs 10]
