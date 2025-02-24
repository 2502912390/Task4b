import torch
import torch.nn as nn
import torchvision

class CRNN(nn.Module):
    def __init__(self, classes_num, cnn_filters, _dropout_rate, rnn_hid=256,in_channels=1):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)

        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)

        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)

        self.pool1 = nn.MaxPool2d(kernel_size=(4, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(8, 1))

        self.dropout = nn.Dropout(_dropout_rate)

        self.gru1 = nn.GRU(128, rnn_hid, bidirectional=True, batch_first=True)

        self.nclass = classes_num

    def forward(self, input):

        x = self.conv1(input)#(bs 1 128 626)->(bs 128 128 626)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)#(bs 128 128 626)->(bs 128 32 313)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)#(bs 128 32 313)->(bs 128 8 156)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)#(bs 128 8 156)->(bs 128 1 156)
        x = self.dropout(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1) #(bs 128 1 156)->(bs 156 128)

        # Bidirectional layer
        recurrent, _ = self.gru1(x)#(bs 156 128)->(2 156 256)

        return  recurrent
    

class CRNN_A(nn.Module):
    def __init__(self, classes_num, cnn_filters=128, dropout=0.3):
        super().__init__()
        # 深层卷积模块
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, cnn_filters, (5,5), padding='same'),  # 大卷积核捕获低频特征
            nn.BatchNorm2d(cnn_filters),
            nn.MaxPool2d((4,2)),  # 增强时域压缩
            
            DepthwiseSeparableConv(cnn_filters, cnn_filters*2, (3,3)),  # 深度可分离卷积
            nn.MaxPool2d((4,2)),
            
            DepthwiseSeparableConv(cnn_filters*2, cnn_filters*4, (3,3)),
            nn.AdaptiveAvgPool2d((None, 1))  # 保持时间维度
        )
        
        # 简化时序建模
        self.gru = nn.GRU(cnn_filters*4, 64, bidirectional=True)
        self.classifier = nn.Linear(128, classes_num)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze(-1).permute(0,2,1)
        x, _ = self.gru(x)
        return self.classifier(x[:, -1, :])
    

class CRNN_B(nn.Module):
    def __init__(self, classes_num, cnn_filters=64, dropout=0.5):
        super().__init__()
        # 多尺度特征提取
        self.conv_block = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, cnn_filters, (3,3)), 
                nn.BatchNorm2d(cnn_filters),
                nn.ReLU(),
                nn.MaxPool2d((2,2))
            ),
            nn.Sequential(
                nn.Conv2d(1, cnn_filters, (5,5), dilation=2),
                nn.BatchNorm2d(cnn_filters),
                nn.ReLU(),
                nn.MaxPool2d((2,2))
            )
        ])
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # 时序建模
        self.gru = nn.GRU(256, 128, bidirectional=True)

    def forward(self, x):
        x1 = self.conv_block[0](x)
        x2 = self.conv_block[1](x)
        x = torch.cat([x1, x2], dim=1)
        attn_weights = self.attention(x)
        x = (x * attn_weights).sum(dim=2)
        x, _ = self.gru(x)
        return x
    
class CRNN_C(nn.Module):
    def __init__(self, classes_num, cnn_filters=64):
        super().__init__()
        # 动态特征提取
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, cnn_filters, (3,3), stride=(1,2)),  # 频率轴下采样
            nn.InstanceNorm2d(cnn_filters),
            nn.GELU(),
            
            nn.Conv2d(cnn_filters, cnn_filters*2, (3,3), groups=cnn_filters),  # 分组卷积
            nn.InstanceNorm2d(cnn_filters*2),
            nn.GELU()
        )
        
        # Transformer时序建模
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024
            ), num_layers=3
        )
        
        # 多尺度分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Linear(128, classes_num)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0,3,2,1).flatten(2)
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))

class CRNN_D(nn.Module):
    def __init__(self, classes_num, cnn_filters=64):
        super().__init__()
        # 高频特征提取
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, cnn_filters, (1,7)),  # 宽频带卷积
            nn.BatchNorm2d(cnn_filters),
            nn.Hardswish(),
            
            nn.Conv2d(cnn_filters, cnn_filters*2, (7,1)),  # 时域卷积
            nn.BatchNorm2d(cnn_filters*2),
            nn.Hardswish(),
            
            nn.AdaptiveMaxPool2d((None,1))  # 保持时间分辨率
        )
        
        # 时域增强模块
        self.temporal_block = nn.Sequential(
            nn.Conv1d(128, 64, 5),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(1024, classes_num)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze(-1)
        return self.temporal_block(x)