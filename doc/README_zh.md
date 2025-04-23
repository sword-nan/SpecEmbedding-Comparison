## SpecEmbedding

SpecEmbedding 是一个专注于 MS/MS 嵌入的模型，它结合正弦嵌入和监督对比学习策略进行训练，在化合物鉴定和检索任务中展现了卓越的性能。

模型的训练与测试基于 MSBERT 清理后的数据集，数据来源于 GNPS、MoNA 和 MTBLS1572。注意到原始数据中仍存在少量 SMILES 序列错误，我们进一步对其进行了清洗以提高数据质量。经过我们处理的数据现已存储于 Zenodo 数据库中，其中不仅包含了清洗后的数据，还有我们针对查询集和参考集进行 10 次随机划分的结果。

为了确保实验结果的公正性和可靠性，我们在处理数据集时保留了 MSBERT 对训练集的原始划分，即训练集和 MSBERT 保持一致，仅对测试集进行了多次随机划分，并最终在这些划分上计算平均值和方差以得出结论。

此外，关于超参数搜索、模型训练过程以及消融实验的详细信息，连同各模型及其表现均已在 Zenodo 数据库中公开。

### 1. 环境配置

Linux Ubuntu 20.04
python 3.12
torch 2.6.0 + cu124

### 2. 示例

#### 2.1 计算 query 和 reference 的余弦相似度矩阵
```python
import torch

from utils import embedding
from train import ModelTester
from data import Tokenizer
from model import SiameseModel
from utils import read_raw_spectra, cosine_similarity

q = read_raw_spectra("./q.msp")
r = read_raw_spectra("./r.msp")

tokenizer = Tokenizer(100, True)
device = "cpu"
model = SiameseModel(
    embedding_dim=512,
    n_head=16,
    n_layer=4,
    dim_feedward=512,
    dim_target=512,
    feedward_activation="selu"
)
model_state = torch.load("./model.ckpt", device)
model.load_state_dict(model_state)
tester = ModelTester(model, device, True)
q, _ = embedding(tester, tokenizer, 512, q, True)
r, _ = embedding(tester, tokenizer, 512, r, True)

cosine_scores = cosine_similarity(q, r)
```

#### 2.2 计算 Top1 的候选化合物

```python
import torch

from utils import embedding
from train import ModelTester
from data import Tokenizer
from model import SiameseModel
from utils import read_raw_spectra, cosine_similarity, top_k_indices

show_progress_bar = False
q_spectra = read_raw_spectra("./q.msp")
r_spectra = read_raw_spectra("./r.msp")

tokenizer = Tokenizer(100, True)
device = "cpu"
model = SiameseModel(
    embedding_dim=512,
    n_head=16,
    n_layer=4,
    dim_feedward=512,
    dim_target=512,
    feedward_activation="selu"
)
model_state = torch.load("./model.ckpt", device)
model.load_state_dict(model_state)
tester = ModelTester(model, device, show_progress_bar)
q, _ = embedding(tester, tokenizer, 512, q_spectra, show_progress_bar)
r, _ = embedding(tester, tokenizer, 512, r_spectra, show_progress_bar)
cosine_scores = cosine_similarity(q, r)

indices = top_k_indices(cosine_scores, 1)
for i, index in enumerate(indices[:, 0]):
    print(f"The {i}-th {q_spectra[i].get('smiles')} spectra most similar compound is {r_spectra[index].get('smiles')}")
```

注明: 在 windows 平台上进行测试时，计算余弦相似度矩阵会出现数值错误的情况，注释掉 @njit 装饰器数值就恢复正常。如果有使用该平台进行测试时需要注释掉 numba 的所有装饰器

### 3. Web 服务

我们也为用户提供了一个 web 服务，每个人都可以通过访问网址 [SpecEmbedding](https://huggingface.co/spaces/xp113280/SpecEmbeeding) 使用。