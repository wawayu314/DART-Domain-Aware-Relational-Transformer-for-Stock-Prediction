# Stock Prediction with Sparse Industry Attention

基于稀疏注意力机制的股票预测系统，专为多市场股票预测而设计。

## 项目简介

本项目实现了一个先进的股票预测模型，结合了：
- **多市场支持**：NYSE、NASDAQ、SP500、A股等不同市场的专门优化
- ** 稀疏注意力机制**：基于行业信息的稀疏注意力，提高计算效率
- **多尺度架构**：通过多尺度卷积捕捉不同时间尺度的模式
- ** 智能参数配置**：根据不同市场自动调整超参数

## 主要特性

###  多市场配置系统
- **NYSE**  - 成熟市场，注重稳定性和排名质量
- **NASDAQ** - 科技股市场，适应快速变化
- **SP500**  - 大盘蓝筹，长期趋势明显

###  稀疏注意力机制
- 基于行业信息的连接模式
- 显著降低计算复杂度
- 保持预测性能的同时提升效率

### 性能评估
- **IC (信息系数)**：衡量预测与真实收益的相关性
- **RIC (排名信息系数)**：评估排名预测质量
- **Precision@10**：前10%股票的预测准确率
- **夏普比率**：风险调整后的收益指标

## 安装说明

### 1. 环境要求
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 数据准备
确保数据集按以下结构组织：
```
dataset/
├── NYSE/
│   ├── eod_data.pkl
│   ├── mask_data.pkl
│   ├── gt_data.pkl
│   ├── price_data.pkl
│   ├── nyse_ticker.csv
│   └── nyse_industry_data.json
├── NASDAQ/
│   └── ...
├── SP500/
│   ├── SP500.npy
│   ├── sp500_ticker.csv
│   └── sp500_industry_data.json
└── A_SHARE/
    └── ...
```

##  使用方法

### 1. 基础训练
```bash
cd src
python train.py
```

### 2. 切换市场
在 `train.py` 中修改市场名称：
```python
market_name = 'SP500'  # 或 'NYSE', 'NASDAQ'
```

### 3. 自定义参数
系统会根据市场自动选择最优参数，也可以手动调整：
```python
# 在对应市场的配置块中修改参数
elif market_name.upper() == 'SP500':
    lookback_length = 32    # 回看长度
    epochs = 60            # 训练轮数
    learning_rate = 0.00002 # 学习率
    alpha = 0.5            # 损失权重
    # ... 其他参数
```

##  项目结构

```
Stock/
├── src/
│   ├── train.py              # 主训练脚本
│   ├── model.py              # 模型定义
│   ├── load_data.py          # 数据加载
│   ├── evaluator.py          # 性能评估
│   └── ...
├── dataset/                  # 数据集目录
├── training_results.xlsx     # 训练结果记录
├── README.md
└── requirements.txt
```





