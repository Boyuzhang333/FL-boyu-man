# 项目进度更新 - Man的部分已完成

## ✅ Man已完成的部分（标签反转攻击）

### 1. 攻击代码实现
**文件**: `client_mal.py` 中的 `train()` 函数
- ✅ 实现了标签反转攻击：`y → (y+1)%10`
- ✅ 50%随机触发概率
- ✅ 支持iid和non_iid_class数据分布
- ✅ 已通过基础测试验证

### 2. 测试文件
- ✅ `test_attack.py` - 单元测试，验证攻击逻辑正确性
- ✅ `experiment_guide.sh` - **Man专用**实验指南，提供标签反转攻击的运行命令

### 3. 环境配置
- ✅ 修复了客户端参数解析（支持node_id 0-4）
- ✅ 验证了Flower框架正常工作
- ✅ CIFAR-10数据集正常加载

## 🎯 Boyu需要完成的部分

### 1. 模型篡改攻击 (Model Poisoning / Gradient Ascent)
在 `client_mal.py` 的 `train()` 函数中实现：
```python
### Type: model_poisoning (Boyu的任务 / Boyu's task)
elif attack_type == 'model_poisoning':
    # TODO: 实现梯度上升攻击
    # 1. 50%概率执行攻击
    # 2. 使用梯度上升而不是梯度下降
    # 3. 参考Man的实现方式
    pass
```

### 2. 防御机制实现
修改 `server.py`，实现两种防御算法：
- **FedMedian**: 坐标中位数聚合
- **FedTrimmedAvg**: 截断平均聚合（去除极端值）

### 3. 实验和分析
- 测试模型篡改攻击在iid/non_iid条件下的效果
- 评估两种防御机制的有效性
- 生成对应的图表和分析报告
- **注意**: Boyu需要创建自己的实验指南来测试模型篡改攻击

## 📋 实验配置要求

**共同配置**：
- 客户端数量：5个
- 恶意客户端数：0, 1, 2, 3
- 数据分布：iid, non_iid_class
- 每配置重复：5次
- 训练轮数：20轮

**Man的责任**：标签反转攻击的8种配置实验
**Boyu的责任**：模型篡改攻击 + 防御机制实验

## 🚀 如何运行和测试

### 快速验证攻击代码
```bash
python test_attack.py  # 验证标签反转攻击
```

### 查看实验指南
```bash
bash experiment_guide.sh  # 显示所有实验配置和命令
```

### 手动运行实验示例（6个终端）
```bash
# Terminal 1: 服务器
python server.py --round 20

# Terminal 2-6: 客户端（示例：1个恶意+4个正常）
python client_mal.py --node_id 0 --n 5 --data_split iid --attack_type label_flipping
python client_mal.py --node_id 1 --n 5 --data_split iid --attack_type none
python client_mal.py --node_id 2 --n 5 --data_split iid --attack_type none
python client_mal.py --node_id 3 --n 5 --data_split iid --attack_type none
python client_mal.py --node_id 4 --n 5 --data_split iid --attack_type none
```

## 🔄 下一步协作

1. **Boyu**: 实现模型篡改攻击和防御机制
2. **Man**: 开始运行标签反转攻击的实验
3. **共同**: 收集实验数据，制作图表，撰写报告

## 📞 注意事项

- **环境一致性**: 确保使用相同的conda环境 `fl-miage`
- **参数兼容性**: 客户端参数已支持node_id 0-4，防止冲突
- **结果保存**: 每次实验后保存log.txt文件，命名格式：`results_mal{X}_{split}_rep{Y}.txt`

---
*更新时间: 2025-11-30*  
*Man的标签反转攻击部分已完成，可以独立开始实验*  
*Boyu可以并行开发他的模型篡改攻击和防御机制*
