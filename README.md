# RobTest-Demo

原材料：(1) 数据集, (2) 训练好的模型。

1. 编写自定义模型加载接口

```python
def load_jigsaw_model():
    evaluated_model = torch.load("jigsaw-roberta-large",map_location=torch.device('cpu'))
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    return evaluated_model, tokenizer

def load_agnews_model():
    evaluated_model = torch.load("ag_newsroberta-large",map_location=torch.device('cpu'))
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    return evaluated_model, tokenizer

```

2. 编写自定义数据加载接口

```python
def read_jigsaw(base_path):
    def read_data(file_path):
        data = pd.read_csv(file_path).values.tolist()
        processed_data = []
        for item in data:
            processed_data.append((item[0].strip(),item[1]))  # [sent,label]
        return processed_data   # ([sent,label],[sent,label]...)
    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')
    train, test = read_data(train_path),  read_data(test_path)
    return train, test
```

3. 运行

   （1） 解压样例数据

   ``` 
   cd RobTest-Demo/data
   unzip sst2.zip
   unzip jigsaw.zip
   unzip agnews.zip
   ```

   （2） 终端运行

   ```
   cd RobTest-Demo
   python robtest.py --mode score --attacker typo --data agnews  --dis_type word --choice both --victim_model textattack/roberta-base-ag-news
   ```

   - mode:  default='score', choices=['**rule', 'score'**, 'decision', 'gradient'])

   - degree, default=-1 # range from [0, 1]  **0-0.6**

   - attacker, default='typo', choices=**['typo','glyph', 'contextual', 'inflection', 'synonym','distraction'])**

   - aug_num, default=100  # 每条句子生成的对抗样本 

   - data，default='sst2'  # **改成你所定义的数据集名称**

   - choice, default='both', choices=['average', 'worst',**'both'**]  

   - dis_type, default='char'，choices=['char', 'word']     

     #'typo','glyph'时可选char/word，其他时候都选word。

   - victim_model, default='roberta-large'  # **改成你所定义的模型名称**

4. 保存成csv

   第一列：degree

   第二列（第三列）：不同degree下的accuracy

   同时会有一个加权score 

   

**预先准备：**

1. 安装相关库： transformers~=4.14.1 datasets~=1.17.0  OpenHowNet==1.0 。。。。

2. 安装nltk的相关工具包

3. GPU 

   os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）

   os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  #（代表仅使用第0，1号GPU）










