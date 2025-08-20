import torch
import torch.nn as nn
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class MultiKernelConv1DTrans(nn.Module):
    def __init__(
        self,
        in_channels=768,
        out_channels=768,
        cls_num = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
        self.conv2 = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    padding=2,
                )
        self.conv3 = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=7,
                    padding=3,
                )
        self.norm = nn.LayerNorm(out_channels)
        self.active = nn.ReLU()
        
        # self.final_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.translayer = TransLayer(dim=out_channels)
        
        # self.head = nn.Sequential(
        #             nn.Linear(in_channels, in_channels),
        #             nn.ReLU(),
        #             nn.Linear(in_channels, cls_num)
        #         )
    
    def forward(self, x):
        # 原始输入（用于残差连接）
        if len(x.shape)== 2:
            x = x.unsqueeze(0)
        input = x.permute(0, 2, 1)
        
        # 多核卷积处理
        output1 = self.conv1(input)
        output1 = self.active(self.norm(output1.permute(0, 2, 1)))
        output2 = self.conv2(input)
        output2 = self.active(self.norm(output2.permute(0, 2, 1)))
        output3 = self.conv3(input)
        output3 = self.active(self.norm(output3.permute(0, 2, 1)))
        output = output1 + output2 + output3 + x
        
                # 残差卷积
                # output = self.final_conv(output) + output
                # output = self.active(self.norm(output.permute(0, 2, 1)))
        
        ## trans-layer
        output = self.translayer(output)
        output = self.norm(output)
       
        # 全连接层
        # output = self.head(output)
        return output

class AttentionAgg(nn.Module):
    def __init__(self, dim=768, cls_num = 3):
        super().__init__()
        
        self.attn = nn.Sequential(
            nn.Linear(dim, 1),  # 为每个patch生成重要性分数
            nn.Softmax(dim=0)   # 在patch维度归一化
        )
        self.head = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, cls_num)
                )

    def forward(self, x):
        # x: [N, 768]
        attn_weights = self.attn(x)  # [N, 1]
        slide_emb = torch.sum(attn_weights * x, dim=0, keepdim=True)  # [1, 768]
        slide_prob = self.head(slide_emb)
        
        return slide_prob


class ConvAttentionAgg(nn.Module):
    def __init__(self, dim=768, cls_num = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=1,
                )
        self.conv2 = nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=5,
                    padding=2,
                )
        self.conv3 = nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=7,
                    padding=3,
                )
        self.norm = nn.LayerNorm(dim)
        self.active = nn.ReLU()
        
        self.attn = nn.Sequential(
            nn.Linear(dim, 1),  # 为每个patch生成重要性分数
            nn.Softmax(dim=0)   # 在patch维度归一化
        )
        self.head = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, cls_num)
                )

    def forward(self, x):
        
        if len(x.shape)== 2:
            x = x.unsqueeze(0)
        input = x.permute(0, 2, 1)
        
        # 多核卷积处理
        output1 = self.conv1(input)
        output1 = self.active(self.norm(output1.permute(0, 2, 1)))
        output2 = self.conv2(input)
        output2 = self.active(self.norm(output2.permute(0, 2, 1)))
        output3 = self.conv3(input)
        output3 = self.active(self.norm(output3.permute(0, 2, 1)))
        output = output1 + output2 + output3 + x
        
        output = output.squeeze()
        # x: [N, 768]
        attn_weights = self.attn(output)  # [N, 1]
        slide_emb = torch.sum(attn_weights * output, dim=0, keepdim=True)  # [1, 768]
        slide_prob = self.head(slide_emb)
        
        return slide_prob
    

class ConvTransAttentionAgg(nn.Module):
    def __init__(self, dim=768, cls_num = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=1,
                )
        self.conv2 = nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=5,
                    padding=2,
                )
        self.conv3 = nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=7,
                    padding=3,
                )
        self.norm = nn.LayerNorm(dim)
        self.active = nn.ReLU()
        
        self.translayer = TransLayer(dim=dim)
        
        self.attn = nn.Sequential(
            nn.Linear(dim, 1),  # 为每个patch生成重要性分数
            nn.Softmax(dim=0)   # 在patch维度归一化
        )
        self.head = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, cls_num)
                )

    def forward(self, x):
        
        if len(x.shape)== 2:
            x = x.unsqueeze(0)
        input = x.permute(0, 2, 1)
        
        # 多核卷积处理
        output1 = self.conv1(input)
        output1 = self.active(self.norm(output1.permute(0, 2, 1)))
        output2 = self.conv2(input)
        output2 = self.active(self.norm(output2.permute(0, 2, 1)))
        output3 = self.conv3(input)
        output3 = self.active(self.norm(output3.permute(0, 2, 1)))
        output = output1 + output2 + output3 + x
        
        output = self.translayer(output)
        output = self.norm(output)
        
        output = output.squeeze()
        # x: [N, 768]
        attn_weights = self.attn(output)  # [N, 1]
        slide_emb = torch.sum(attn_weights * output, dim=0, keepdim=True)  # [1, 768]
        slide_prob = self.head(slide_emb)
        
        return slide_prob
