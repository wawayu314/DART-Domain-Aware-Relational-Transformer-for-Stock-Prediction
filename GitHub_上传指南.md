# 如何将DART项目上传到GitHub

## 项目信息
- **项目名称**: DART Domain-Aware Relational Transformer for Stock Prediction
- **GitHub仓库**: https://github.com/wawayu314/DART-Domain-Aware-Relational-Transformer-for-Stock-Prediction.git
- **本地项目路径**: C:\Users\Administrator\Desktop\DART

## 上传步骤

### 1. 打开命令提示符
- 按 `Win + R` 键
- 输入 `cmd` 并按回车
- 导航到项目目录：
  ```
  cd C:\Users\Administrator\Desktop\DART
  ```

### 2. 初始化Git仓库
```
git init
```

### 3. 添加远程仓库
```
git remote add origin https://github.com/wawayu314/DART-Domain-Aware-Relational-Transformer-for-Stock-Prediction.git
```

### 4. 添加所有文件
```
git add .
```

### 5. 提交文件
```
git commit -m "Initial commit: DART Domain-Aware Relational Transformer for Stock Prediction"
```

### 6. 推送到GitHub
```
git push -u origin main
```

## 如果遇到问题

### 如果提示需要认证
- 如果提示输入用户名和密码，请输入你的GitHub用户名和密码
- 如果启用了双因素认证，需要使用个人访问令牌（Personal Access Token）

### 如果推送失败
- 尝试先拉取远程仓库：
  ```
  git pull origin main --allow-unrelated-histories
  ```
- 然后再次推送：
  ```
  git push -u origin main
  ```

## 项目结构
项目包含以下文件：
- `README.md` - 项目说明文档
- `requirements.txt` - Python依赖包
- `src/` - 源代码目录
  - `train.py` - 训练脚本
  - `model.py` - 模型定义
  - `evaluator.py` - 评估脚本
  - `load_data.py` - 数据加载脚本
- `dataset/` - 数据集目录
  - `SP500/` - 标普500数据
  - `NYSE/` - 纽约证券交易所数据
  - `NASDAQ/` - 纳斯达克数据
- `.gitignore` - Git忽略文件配置

## 注意事项
- 项目中已包含 `.gitignore` 文件，会自动排除不必要的文件
- 如果数据集文件较大，建议使用 Git LFS 来管理大文件
- 确保你有该GitHub仓库的写入权限 