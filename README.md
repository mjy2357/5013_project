# Acknowledgement

This repo is  for the 5013 course project. Thanks professor Chu and TAs for your guidance. 

# 1. Install dependencies

```bash
git clone https://github.com/mjy2357/5013_project.git
conda create -n 5013_project 
conda activate 5013_project
cd 5013_project
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

# 2. Run Experiments

To run these 2 experiments, a GPU card is needed. If the memory is limited, you can use a smaller batch size in these 2 .py files.

```bash
python 5013_CNN.py
```

```bash
python ResNet.py
```

