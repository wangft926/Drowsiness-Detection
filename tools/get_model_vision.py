from torchviz import make_dot
import torch
from train0_3_loss import DualInputModel  # 导入你的模型
# 示例输入
geo_input = torch.randn(8, 30, 68)  # (B, T, F)
face_input = torch.randn(8, 30, 3, 112, 112)  # (B, T, C, H, W)

model = DualInputModel(config={'num_features': 68, 'num_classes': 2})
out = model(geo_input, face_input)

# 生成图
dot = make_dot(out, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('dual_input_model', view=False)