# 六种大语言模型在放射医学技术专业知识考核中的统计分析代码与数据

本仓库用于支持研究论文中表2、表3、表4及相关图表的统计分析与结果复现。仓库包含经过整理的分析数据（Excel）、Python分析脚本，以及相应的统计分析流程说明。

**重要说明：** 本仓库仅分享用于统计计算的非识别性分析数据，不公开原始考试题目题干、备选项、标准答案及参考解析。请勿将受版权保护的原始考试材料上传至本仓库。

## 1. 文件结构

```text
.
├── README.md
├── analyze_table2.py
├── analyze_table3.py
├── analyze_table4.py
├── data_table2.xlsx
├── data_table3.xlsx
└── data_table4.xlsx
```

其中，`analyze_table4.py` 为不读取、不导出原题题干及标准答案的公开版本；对应的 `data_table4.xlsx` 也应仅包含下文列出的分析字段。

## 2. 数据说明

### 2.1 `data_table2.xlsx`：客观答题准确率数据

该文件用于六种大语言模型的客观答题准确率分析。每行代表一道题目的模型作答记录，不包含完整题干、备选项或参考解析。

必需字段如下：

| 字段 | 含义 |
|---|---|
| `ID` | 每条题目记录的唯一标识，用于混合效应模型中的随机截距分组 |
| `Question_ID` | 题目编号；可能在不同知识领域或题型中重复，不单独作为随机效应分组变量 |
| `Level` | 专业层级：`Junior Radiographer`、`Radiographer`、`Radiographer-in-charge` |
| `Subject` | 知识领域：`Basic Knowledge`、`Relevant Professional Knowledge`、`Professional Knowledge`、`Professional Practice Skills` |
| `Question_Type` | 题型标识，当前分析使用 `A` 或 `B` |
| 六个模型列 | 模型作答是否正确，使用二分类编码：`1`表示正确，`0`表示错误 |

六个模型列名必须与脚本一致：`Kimi K2`、`DeepSeek R1`、`Qwen3-Max`、`Doubao 1.6`、`gemini flash 2.5`、`chatgpt 4o-mini`。

### 2.2 `data_table3.xlsx`：主观解释质量评分数据

该文件用于两名评价者对六种模型生成解释的Likert评分分析。每行代表一道接受主观评价的题目，不包含题干、模型解释文本或参考解析。

必需字段包括：

- `Question_ID`：题目唯一编号，每行不得重复；
- `Level`：专业层级；
- `Subject`：知识领域；
- 两名评价者分别对六种模型的评分，共12列。

评分列名称由评价者前缀与模型名称组成，例如：`Rater1_Kimi K2`、`Rater2_Kimi K2`。所有评分必须为1–5的整数，其中1分最低、5分最高。层级、知识领域及模型名称应使用脚本中定义的英文标签，不要自行更改拼写或大小写。

### 2.3 `data_table4.xlsx`：基线高错误题目的探索性比较数据

该文件用于分析2025年基线测试中六种模型均答错或仅一种模型答对的高错误题目，并比较2025年与2026年模型组在这些题目中的描述性表现。当前分析数据集包含147道题目：74道题目在2025年基线测试中为0/6模型答对，73道题目为1/6模型答对。

公开数据文件仅需包含以下字段：

| 字段 | 含义 |
|---|---|
| `ID` | 题目记录的唯一编号 |
| `Subject` | 知识领域标签 |
| `Classification` | 原始类别标签 |
| `2025KIMI`、`2025Deepseek`、`2025Qwen`、`2025Doubao`、`2025Gemini`、`2025ChatGPT` | 2025年六种模型的二分类作答结果 |
| `2026kimi`、`2026deepseek`、`2026Qwen`、`2026Doubao`、`2026gemini`、`2026chatgpt` | 2026年六种模型的二分类作答结果 |

模型作答结果使用二分类编码：`1`表示正确，`0`表示错误。`Classification`须使用脚本中预设的原始类别名称。脚本会依据预先设定的规则将题目重新归入五个知识类别，并输出分类调整记录及理由。

**请勿在公开版 `data_table4.xlsx` 中加入 `Question stem`、`Answer`、原题选项或解析等字段。** 公开版 `analyze_table4.py` 已调整为不要求、不读取和不导出题干及标准答案，仅使用题目ID、知识领域、分类和模型二分类结果完成统计分析。

## 3. 分析内容

### 3.1 表2：客观答题准确率

`analyze_table2.py` 主要执行以下分析：

1. 计算六种模型在不同专业层级和知识领域中的答题准确率及95% Wilson置信区间；
2. 在各专业层级 × 知识领域分层内使用Cochran's Q检验比较六种模型的配对二分类结果，并对12个总体检验进行Holm校正；
3. 使用精确McNemar检验进行模型间两两比较，并在各分层内对15组比较进行Holm校正；
4. 使用二项分布混合效应逻辑回归模型分析模型间差异，随机截距以唯一题目记录ID为分组变量；
5. 对A型题进行敏感性分析。

混合效应模型通过 `statsmodels` 的 `BinomialBayesMixedGLM` 和变分贝叶斯方法拟合。相关区间为近似95%后验可信区间，并非传统最大似然估计GLMM的95%置信区间。B型题缺少共同题干/病例的聚类标识，因此无法在主分析中显式建模共同题干聚类；脚本提供A型题敏感性分析。

### 3.2 表3：主观解释质量与评价者间一致性

`analyze_table3.py` 主要执行以下分析：

1. 按评价者、专业层级和知识领域分别使用Friedman检验比较六种模型的评分；
2. 对每名评价者的12个总体检验进行Holm校正；
3. 使用配对Wilcoxon符号秩检验进行模型间两两比较，并在相应分层内进行Holm校正；
4. 计算评分中位数、四分位数间距、Kendall's W及1–5分评分分布；
5. 统计低分比例（评分≤2）；
6. 计算评价者间绝对一致性ICC(A,1)，以及线性和二次加权Cohen's kappa，并通过题目层面Bootstrap估计95%置信区间。

Bootstrap重复次数可通过 `--bootstrap` 参数设置，默认值为1000次。

### 3.3 表4：基线高错误题目的探索性分析

`analyze_table4.py` 主要执行以下分析：

1. 检查题目是否符合2025年基线筛选条件（六种模型中0或1种答对）；
2. 根据预设规则将原始类别重新归并为五个知识类别，并记录类别调整及原因；
3. 计算各类别2025年与2026年模型组的总体准确率；
4. 以题目为单位进行Bootstrap重抽样，估计准确率及准确率差值的95%置信区间；
5. 统计每道题在各时间点由0–6种模型答对的分布。

默认Bootstrap重复次数为10000次。由于题目是根据2025年基线表现筛选，且两个时间点的模型版本并非严格对应，因此该分析仅用于特定高错误题目集合中的探索性描述，不用于推断单个模型的纵向能力变化或模型进化。脚本不计算跨年度推断性P值。

## 4. 软件环境与依赖

建议使用Python 3.10或更高版本，并在独立虚拟环境中运行。

表2脚本使用的主要第三方依赖：

```bash
python -m pip install numpy pandas scipy matplotlib statsmodels openpyxl
```

表3和表4脚本还调用 `artifact_tool` 中的 `Blob`、`SpreadsheetFile` 和 `Workbook` 接口，用于Excel文件导入及结果工作簿导出。该模块不是Python标准库的一部分；运行这两个脚本前，请确认本地环境已配置与脚本API兼容的 `artifact_tool`。本仓库目前未附带该依赖的源码或安装包。

建议记录实际运行时的Python及依赖包版本，以便后续复现和核对结果。

## 5. 运行方法

请将三个Excel数据文件与对应Python脚本放在同一目录。在该目录下执行：

**表2：**

```bash
python analyze_table2.py --input data_table2.xlsx --output_dir table2_outputs
```

**表3：**

```bash
python analyze_table3.py --input data_table3.xlsx --output_dir table3_outputs --bootstrap 1000
```

**表4：**

```bash
python analyze_table4.py --input data_table4.xlsx --output_dir table4_outputs --bootstrap 10000
```

各脚本会自动创建输出目录。若使用默认参数，也可以直接运行脚本；以上命令显式指定输入文件、输出目录及Bootstrap次数，便于复现。

## 6. 主要输出文件

### 表2输出

- `Table2_Statistical_Results.xlsx`：主要统计结果及诊断信息；
- 多个CSV文件：包括总体准确率、Cochran's Q、McNemar两两比较、GLMM结果、A型题敏感性分析和数据质量检查；
- `Figure_Table2_Accuracy_DotWhisker.png/.pdf`：准确率及Wilson区间图；
- `FigureS1_GLMM_OR_Forest.png/.pdf`：GLMM比值比森林图。

### 表3输出

- `Table3_Statistical_Results.xlsx`：评分分布、Friedman检验、Wilcoxon检验及评价者间一致性结果；
- `Table3_Main.csv`、`Friedman_Omnibus.csv`、`Wilcoxon_Pairwise.csv`；
- `Score_Distribution.csv`、`Low_Score_Rate.csv`、`Interrater_Reliability.csv`、`Data_QC.csv`；
- `Figure2_Subjective_Score_Distribution.png/.pdf`：主观评分分布图；
- `FigureS2_Low_Score_Heatmap.png/.pdf`：低分比例热图。

### 表4输出

- `Table4_Revised_Analysis.xlsx`：重新分类后的统计结果及数据质量检查；
- `Table4_Revised_Main.csv`：五个修订类别的主要结果；
- `Table4_Original_Category_Summary.csv`：原始类别汇总；
- `FigureS3_ItemCorrectCount_Data.csv`：各类别中答对模型数量的分布数据；
- `Classification_Changes.csv`：类别调整记录及理由；
- `Item_Level_Revised_Classification.csv`：逐题分类及正确模型数量，不含原题题干和标准答案；
- `Figure3_HighError_RevisedCategories.png/.pdf`：2025与2026年模型组准确率比较图；
- `FigureS3_ItemCorrectCount_RevisedCategories.png/.pdf`：逐题答对模型数量分布图；
- `FigureS4_Category_Counts_Original_vs_Revised.png/.pdf`：原始与修订类别数量比较图。

## 7. 结果解释与复现注意事项

1. 本仓库提供的分析数据为研究统计用途的整理数据，不包含完整考试题目内容。不同脚本的输入数据结构不同，请勿将三个Excel文件的字段混用。
2. 表2的`ID`必须唯一；`Question_ID`可能在不同领域或题型中重复。模型结果必须完整且仅为0或1。
3. 表3的`Question_ID`在输入表中必须唯一；12个评分列均须为1–5的整数，不得存在缺失值。
4. 表4必须严格满足147道高错误题目的筛选条件，且每道题在2025年基线模型组中只能有0或1个模型答对。六种模型在两个时间点的结果均须为0或1。
5. 表4的年份比较受基线筛选和模型版本不完全匹配的限制，应解释为探索性描述结果，而非单个模型的纵向性能提升。
6. 模型名称仅代表分析数据表中的列标签；论文中应以实际API调用记录核对模型的准确版本及调用时间。
7. 若修改数据、分类规则、模型列名或统计方法，应同步更新分析脚本、结果文件及论文中的统计学方法描述。

## 8. 数据与代码使用说明

本仓库的数据仅用于论文相关统计分析和结果复现。原始考试题目、备选项、标准答案及参考解析可能受到出版物版权保护，因此未在本仓库公开。使用者不得根据本仓库数据尝试重构或重新发布受版权保护的考试材料。

代码用于复现本研究所采用的统计流程，不构成对模型优劣、临床适用性或专业胜任力的独立认证。使用者应结合研究设计、数据结构和统计假设审慎解释分析结果。

如引用本仓库，请在论文或相关成果中注明仓库地址、访问日期及所使用的版本或提交记录。
