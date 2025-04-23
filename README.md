## SpecEmbedding

Chinese Version: [SpecEmbedding](./doc/README_zh.md)

SpecEmbedding is a model specifically designed for MS/MS embedding. It combines sinusoidal embedding with supervised contrastive learning strategies during training and has demonstrated outstanding performance in compound identification and retrieval tasks.

The model's training and testing are based on the dataset cleaned by MSBERT, sourced from GNPS, MoNA, and MTBLS1572. Noticing that there were still a small number of SMILES sequence errors in the raw data, we further cleaned it to improve data quality. The processed data is now stored in the Zenodo database, which includes not only the cleaned data but also 10 random splits of the query set and reference set.

To ensure fairness and reliability of the experimental results, we retained the original split of the training set used by MSBERT when processing the dataset. Specifically, the training set remained consistent with MSBERT, while the test set was randomly split multiple times. The final conclusions were drawn by calculating the mean and variance across these splits.

Additionally, detailed information regarding hyperparameter search, model training processes, ablation experiments, as well as all models and their performances, have been made publicly available in the Zenodo database.

### 1. Environment Configuration
Linux Ubuntu 20.04
Python 3.12
PyTorch 2.6.0 + CUDA 12.4

### 2. Examples
#### 2.1 Calculate the Cosine Similarity Matrix Between Query and Reference

```python
import torch

from utils import embedding
from train import ModelTester
from data import Tokenizer
from model import SiameseModel
from utils import read_raw_spectra, cosine_similarity

# Load query and reference spectra
q = read_raw_spectra("./q.msp")
r = read_raw_spectra("./r.msp")

# Initialize tokenizer and device
tokenizer = Tokenizer(100, True)
device = "cpu"

# Define the SiameseModel architecture
model = SiameseModel(
    embedding_dim=512,
    n_head=16,
    n_layer=4,
    dim_feedward=512,
    dim_target=512,
    feedward_activation="selu"
)

# Load the pre-trained model state
model_state = torch.load("./model.ckpt", device)
model.load_state_dict(model_state)

# Initialize the ModelTester
tester = ModelTester(model, device, True)

# Generate embeddings for query and reference spectra
q, _ = embedding(tester, tokenizer, 512, q, True)
r, _ = embedding(tester, tokenizer, 512, r, True)

# Compute the cosine similarity matrix
cosine_scores = cosine_similarity(q, r)
```

#### 2.2 Calculate the Top-1 Candidate Compounds

```python
import torch

from utils import embedding
from train import ModelTester
from data import Tokenizer
from model import SiameseModel
from utils import read_raw_spectra, cosine_similarity, top_k_indices

# Disable progress bar for simplicity
show_progress_bar = False

# Load query and reference spectra
q_spectra = read_raw_spectra("./q.msp")
r_spectra = read_raw_spectra("./r.msp")

# Initialize tokenizer and device
tokenizer = Tokenizer(100, True)
device = "cpu"

# Define the SiameseModel architecture
model = SiameseModel(
    embedding_dim=512,
    n_head=16,
    n_layer=4,
    dim_feedward=512,
    dim_target=512,
    feedward_activation="selu"
)

# Load the pre-trained model state
model_state = torch.load("./model.ckpt", device)
model.load_state_dict(model_state)

# Initialize the ModelTester
tester = ModelTester(model, device, show_progress_bar)

# Generate embeddings for query and reference spectra
q, _ = embedding(tester, tokenizer, 512, q_spectra, show_progress_bar)
r, _ = embedding(tester, tokenizer, 512, r_spectra, show_progress_bar)

# Compute the cosine similarity matrix
cosine_scores = cosine_similarity(q, r)

# Retrieve the indices of the top-1 candidates
indices = top_k_indices(cosine_scores, 1)
for i, index in enumerate(indices[:, 0]):
    print(f"The {i}-th spectra with SMILES {q_spectra[i].get('smiles')} most similar compound is {r_spectra[index].get('smiles')}")
```

Note: When testing on the Windows platform, numerical errors may occur when computing the cosine similarity matrix. Commenting out the @njit decorator resolves this issue. If testing on this platform, ensure to comment out all Numba decorators.

### 3.Web Server

We also provide a web server for the users. Everyone can visit the website [SpecEmbedding](https://huggingface.co/spaces/xp113280/SpecEmbeeding)