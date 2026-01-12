import torch
from torchsummary import summary  # 可选，先查看输入输出维度
import hiddenlayer as hl
from train0_3_loss import DualInputModel  # 导入你的模型

# 1. 定义模型和示例输入
config = {
    "num_features": 10,  # 根据你的实际配置填写
    "num_classes": 3     # 替换为你的类别数
}
model = DualInputModel(config)

# 2. 定义示例输入（匹配模型的输入维度）
# geo_input: (B, F, T) → 假设B=2, F=10, T=5
geo_input = torch.randn(2, 10, 5)
# face_input: (B, T, C, H, W) → 假设B=2, T=5, C=3, H=224, W=224
face_input = torch.randn(2, 5, 3, 224, 224)

# 3. 用HiddenLayer构建可视化流程
# 定义转换规则：简化模块名称（避免显示过于冗长的类名）
transforms = [
    hl.transforms.Fold("CrossAttention", "CrossAttention"),  # 自定义模块名
    hl.transforms.Fold("FrameFeatureExtractor", "FrameFeatureExtractor\n(MobileNet)"),
    hl.transforms.Fold("GatedFusion", "GatedFusion"),
    hl.transforms.Fold("nn.GRU", "BiGRU"),  # 简化GRU显示
    hl.transforms.Fold("nn.Linear", "Linear"),
    hl.transforms.Fold("nn.Sequential", "Classifier"),  # 合并分类器模块
]

# 4. 生成可视化图
graph = hl.build_graph(
    model,
    (geo_input, face_input),  # 传入示例输入（元组形式，对应模型的两个输入）
    transforms=transforms,
    input_names=["Geo Input\n(B, F, T)", "Face Input\n(B, T, C, H, W)"]  # 标注输入名称
)

# 5. 保存为图片（PDF格式适合论文，清晰度高）
graph.save("model_structure.pdf", format="pdf")
graph.show()  # 实时查看