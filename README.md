
### 阶段 1：项目启动与数据探索

#### 1.1 项目目标确定
- 明确项目目标：预测银行账户欺诈。
- 确定评估指标：如准确率、精确率、召回率、F1分数、AUC-ROC曲线等。重点关注少数类的recall。

#### 1.2 数据获取与初步分析
- 数据加载与查看：初步查看数据结构和基本统计信息。
- 数据清洗：处理缺失值、重复值和异常值。

#### 1.3 数据可视化
- 进行数据分布、类别分布和特征关系的可视化。
- 分析欺诈和非欺诈样本的差异。

### 阶段 2：特征工程与数据预处理

#### 2.1 特征选择与创建
- 特征选择：使用相关性分析、特征重要性等方法选择重要特征。
- 特征创建：根据已有特征创建新的特征（如时间特征、交互特征等）。

#### 2.2 数据编码与标准化
- 类别变量编码：使用独热编码、标签编码等方法对类别变量进行编码。
- 数值变量标准化：对数值变量进行标准化或归一化处理。

#### 2.4 不平衡数据的处理
- 过采样：使用 SMOTE 等方法对少数类样本进行过采样。
- 欠采样：对多数类样本进行欠采样。

### 阶段 3：模型选择与训练

#### 3.1 基准模型建立
- 使用简单模型（如逻辑回归、决策树）建立基准模型，评估初步效果。

#### 3.2 基于决策树的方法
- 训练随机森林模型：调整参数，使用交叉验证优化模型性能。
- 训练 XGBoost 模型：进行超参数调优，评估模型效果。

#### 3.3 神经网络模型
- 构建基础神经网络模型：设计简单的神经网络架构，训练并评估模型。
- 尝试不同的神经网络架构：如将神经网络分别应用于每个输入特征。

### 阶段 4：高级方法探索
- 尝试GNN（不知道是否可行）
- 尝试构建贝叶斯神经网络，评估其在不确定性处理和预测效果上的表现。

#### 4.2 结合现代大型语言模型
- 尝试使用预训练的大型语言模型（如 BERT、GPT）进行特征提取或作为特征的一部分。

### 阶段 5：模型评估与优化

#### 5.1 模型评估
- 统一评估所有模型：使用验证集评估各个模型的性能，记录各项评估指标。主要关注Recall
- 绘制 ROC 曲线、混淆矩阵等可视化结果，比较不同模型的效果。
- 对多个数据集变种进行建立模型和类似的评估，比较不同模型的泛化能力。

#### 5.2 模型优化
- 进行超参数调优：使用网格搜索或随机搜索进行全面的超参数优化。
- 集成学习：尝试将多个模型进行集成，如软投票、硬投票、堆叠模型等方法。



### Stage 1: Project Initiation and Data Exploration

#### 1.1 Project Objective Definition
- Define the project objective: to predict bank account fraud.
- Identify evaluation metrics: e.g., accuracy, precision, recall, F1 score, AUC-ROC curve, etc. Focus on few categories of recall.

#### 1.2 Data Acquisition and Preliminary Analysis
- Data loading and viewing: preliminary view of data structure and basic statistical information.
- Data cleaning: deal with missing values, duplicate values and outliers.

#### 1.3 Data Visualisation
- Perform visualisation of data distribution, category distribution and feature relationships.
- Analyse differences between fraudulent and non-fraudulent samples.

### Stage 2: Feature Engineering and Data Preprocessing

#### 2.1 Feature Selection and Creation
- Feature Selection: Select important features using correlation analysis, feature importance and other methods.
- Feature Creation: Create new features (e.g., temporal features, interaction features, etc.) based on existing features.

#### 2.2 Data Coding and Normalisation
- Coding of categorical variables: coding of categorical variables using methods such as unique heat coding and label coding.
- Numeric variable standardisation: standardisation or normalisation of numeric variables.

#### 2.4 Treatment of unbalanced data
- Oversampling: oversampling of minority category samples using methods such as SMOTE.
- Under-sampling: Under-sampling of the majority of the classes.

### Stage 3: Model Selection and Training

#### 3.1 Benchmark model building
- Use simple models (e.g., logistic regression, decision trees) to build a benchmark model and evaluate the initial results.

#### 3.2 Decision Tree Based Approach
- Train Random Forest model: adjust parameters and use cross-validation to optimise model performance.
- Train the XGBoost model: perform hyper-parameter tuning and evaluate model effectiveness.

#### 3.3 Neural Network Models
- Build a basic neural network model: design a simple neural network architecture, train and evaluate the model.
- Experiment with different neural network architectures: e.g. apply neural networks to each input feature separately.

### Stage 4: Exploring Advanced Methods
- Try GNN (not sure if it is feasible)
- Attempt to construct a Bayesian neural network and evaluate its performance in uncertainty handling and predictive effectiveness.

#### 4.2 Combining modern large-scale language models
- Try using pre-trained large language models (e.g. BERT, GPT) for feature extraction or as part of features.

### Stage 5: Model Evaluation and Optimisation

#### 5.1 Model Evaluation
- Evaluate all models uniformly: use validation sets to evaluate the performance of each model and record the evaluation metrics. The main focus is on Recall
- Plot ROC curves, confusion matrices and other visualisation results to compare the effectiveness of different models.
- Model building and similar evaluations are performed on multiple dataset variants to compare the generalisation capabilities of different models.

#### 5.2 Model Optimisation
- Perform hyperparameter tuning: perform full hyperparameter optimisation using grid search or random search.
- Integration learning: attempt to integrate multiple models, e.g. soft voting, hard voting, stacked models, and other methods.

