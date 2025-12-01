# 联邦学习项目进度 - Man的攻击1已完成

## ✅ Man已完成的工作（攻击1 - 标签翻转）

**攻击实现**: 在`client_mal.py`中实现标签翻转攻击 `y → (y+1)%10`，50%概率触发

**实验执行**: 完成40个实验（8种配置 × 5次重复）
- IID和Non-IID数据分布
- 0/1/2/3个恶意客户端场景
- 结果保存在`results1/`目录

**分析可视化**: 攻击效果可视化完成（`plot_attack1.py`和`attack1_results.png`）

## 📁 项目结构与文件说明

### 核心文件（原有）
- `client.py` - 正常联邦学习客户端
- `server.py` - 原始FedAvg服务器 
- `prepare_dataset.py` - CIFAR-10数据加载和分发
- `client_mal.py` - 恶意客户端（包含攻击1实现）

### Man创建的文件
- `serveur_attack1.py` - 攻击1专用服务器，带CSV日志功能
- `attack1_usage_guide.sh` - 攻击1使用指南
- `run_remaining_experiments.sh` - 自动化运行40个实验的脚本
- `plot_attack1.py` - 攻击1结果可视化脚本
- `results1/` - 包含40个CSV结果文件的目录
- `attack1_results.png` - 攻击1效果图表


## 🎯 Boyu待完成任务清单

### 1. 攻击2实现（模型投毒）
**需要修改的文件**: `client_mal.py` - 在`train()`函数中添加模型投毒攻击
```python
elif attack_type == 'model_poisoning':
    # TODO: Implement gradient ascent attack
    # - 50% probability trigger
    # - Use gradient ascent instead of descent
    # - Reference Man's implementation style
```

### 2. 防御机制实现
**需要修改的文件**: `server.py` - 实现两种防御算法：
- **FedMedian**: 坐标中位数聚合
- **FedTrimmedAvg**: 截断平均聚合（移除极值）

### 3. Boyu的实验与分析
**创建类似Man的结构**:
- 创建`serveur_attack2.py`（最好直接用server.py）
- 运行攻击2和防御机制的实验
- 创建结果可视化脚本
- 分析防御有效性

## 🚀 Boyu快速开始

**环境**: 使用现有的`fl-miage` conda环境
```bash
conda activate fl-miage
cd /path/to/projet/FL-boyu-man/Projet
```

**参考Man的工作**:
- 查看`client_mal.py`了解攻击实现模式
- 查看`serveur_attack1.py`了解自动化实验设置
- 使用类似命名：`results2/`存储攻击2结果

**实验配置**: 与Man相同（5个客户端，0/1/2/3个恶意客户端，IID/Non-IID，5次重复）

## 📊 当前实验结果汇总

**Man的攻击1结果**（来自`attack1_results.png`）:
- **IID基准**: ~62%准确率（0个恶意客户端）
- **IID + 1个恶意**: ~53%准确率（下降9%）
- **IID + 2-3个恶意**: ~30%准确率（严重影响）
- **Non-IID基准**: ~50%准确率（0个恶意客户端）
- **Non-IID + 1个恶意**: ~40%准确率（下降10%）
- **Non-IID + 2-3个恶意**: ~15-17%准确率（毁灭性影响）

**关键发现**: Non-IID数据对标签翻转攻击更加脆弱

---
*更新时间: 2025年12月1日*  
*状态: Man的攻击1工作100%完成。Boyu现在可以开始攻击2和防御机制的实现。*

