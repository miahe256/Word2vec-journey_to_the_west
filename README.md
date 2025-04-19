# 中文词向量训练与相似度计算项目

这是一个基于PyTorch实现的中文词向量（Word2Vec）训练和相似度计算项目。项目使用Skip-gram模型训练词向量，并提供了词语相似度计算、词向量运算等功能。

## 项目结构

```
word2vec/
├── word_seg(1).py              # 分词程序
├── word_similarity_torch.py     # 训练模型程序
├── word_similarity_inference.py # 测试相似度程序
├── utils/
│   ├── __init__.py
│   └── files_processing_simple.py
├── journey_to_the_west/        # 西游记数据集
│   ├── source/                 # 原始文本
│   └── segment/               # 分词结果
└── models/                    # 保存训练好的模型
```

## 环境要求

- Python 3.6+
- PyTorch
- jieba
- numpy

可以通过以下命令安装依赖：
```bash
pip install torch numpy jieba
```

## 使用指南

### 1. 数据预处理（分词）
运行分词程序：
```bash
python word_seg\(1\).py
```
这将对 `journey_to_the_west/source` 目录下的文本文件进行分词，结果保存在 `journey_to_the_west/segment` 目录。

### 2. 训练词向量模型
运行训练程序：
```bash
python word_similarity_torch.py
```
训练完成后，模型会保存在 `models/word2vec_torch.pt` 文件中。

主要的超参数包括：
- vector_size：词向量维度（默认100）
- window：上下文窗口大小（默认3）
- min_count：最小词频（默认1）
- epochs：训练轮数（默认5）

### 3. 测试词语相似度
运行测试程序：
```bash
python word_similarity_inference.py
```
这个程序提供了以下功能：
- 计算词语之间的相似度
- 查找与目标词最相似的词
- 支持词向量运算（如：孙悟空 + 唐僧 - 大王）

## 切换数据源（从西游记到三国演义）

如果要将数据源从西游记切换到三国演义，需要进行以下修改：

### 1. 修改 word_seg(1).py
```python
# 修改源文件目录和分词输出目录
source_folder = './three_kingdoms/source'    # 改为三国演义源文本目录
segment_folder = './three_kingdoms/segment'  # 改为三国演义分词输出目录
```

### 2. 修改 word_similarity_torch.py
```python
# 修改训练数据路径
segment_folder = './three_kingdoms/segment'  # 改为三国演义分词目录

# 可选：修改模型保存路径，以区分不同数据集训练的模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'vector_size': 100
}, './models/word2vec_torch_three_kingdoms.pt')  # 改为新的模型文件名
```

### 3. 修改 word_similarity_inference.py
```python
# 修改模型加载路径
model_path = './models/word2vec_torch_three_kingdoms.pt'  # 改为新的模型文件名

# 修改测试词语为三国演义相关的人物或词语
test_words = ['曹操', '刘备', '孔明', '关羽']  # 根据需要更换测试词语
```

### 4. 目录结构调整
1. 创建新的数据目录：
```
mkdir three_kingdoms
mkdir three_kingdoms/source
mkdir three_kingdoms/segment
```

2. 将三国演义的文本文件放入 `three_kingdoms/source` 目录

## 注意事项

1. 确保源文本文件使用UTF-8编码
2. 分词结果会自动保存为UTF-8编码
3. 模型训练时间取决于数据量和设备性能
4. 如果有GPU可用，程序会自动使用GPU加速训练
5. 可以根据需要调整超参数以获得更好的效果

## 常见问题

1. 如果出现词语不在词表中的错误，可能是因为：
   - 该词在训练集中出现次数少于min_count
   - 分词结果中没有该词
   - 词语的编码格式不一致

2. 如果训练速度较慢：
   - 可以减小vector_size的值
   - 可以增大batch_size
   - 可以使用GPU训练

3. 如果内存不足：
   - 可以减小batch_size
   - 可以增大min_count过滤低频词

## 扩展功能

1. 词向量可视化
2. 添加新的词向量运算
3. 支持其他文本数据源
4. 添加模型评估指标

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。 
