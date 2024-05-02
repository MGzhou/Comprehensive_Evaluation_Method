# 综合评价法

实现常见的综合评价法。综合评价法主要分为主观赋权法和客观赋权法。

**【1】主观赋权法**

主观赋权法适合没有样本数据或样本数据非常少的情况。

> **优点**是通过专家制定，可以集合自身经验和知识对不同指标赋予权重，也可以明确地反映出决策者的偏好。
>
> **缺点**是依赖专家判断，不同专家可能会有不同看法，导致一定程度的主观随意性和不确定性。当面临复杂系统的评估时，主观赋权法可能难以准确处理和反映系统中众多因素的关系。

| 算法                                 | 是否实现                        |
| ------------------------------------ | ------------------------------- |
| [AHP 层次分析法](docs/AHP层次分析法.md) | [是](src/subjective_method/AHP.py) |

**【2】客观赋权法**

客观赋权法适合样本数据多的情况。

> 优点是客观性强，减少了主观判断的影响。
>
> 缺点是对数据敏感，对数据的异常值和噪声非常敏感。

客观赋权法是一种以数据驱动为主的权重分配方法，它在确保评价过程的客观性方面具有明显优势，但同时也需要对其局限性保持警惕。在实际应用中，可能需要结合主观赋权法等其他方法来弥补这些不足，以达到更加全面和准确的评价效果。

| 算法                          | 是否实现                          |
| ----------------------------- | --------------------------------- |
| [Critic](docs/Critic法.md)       | [是](src/objective_method/critic.py) |
| [熵权法](docs/熵权法.md)         | [是](src/objective_method/EWM.py)    |
| [变异系数法](docs/变异系数法.md) | [是](src/objective_method/COV.py)    |

## 使用例子

使用Critic方法，算法实现是 `src/critic.py`。用法如 `src/demo.py`

命令行环境切换到src目录运行demo.py

```
cd src

python demo.py
```

代码

```python
import numpy as np
from objective_method.critic import Critic

# 1建立critic对象
critic = Critic()

# 2 样本数据,有6个样本，8个评价指标
data = [[162.519, 1069.70, 55.789, 196.906, 3894.9, 6470.51, 2.635, 1.55],
        [113.073, 763.16, 70.677, 61.947, 1033.9, 7490.32, 1.986, 0.59],
        [245.158, 3962.42, 163.893, 178.782, 536.0, 2288.19, 1.276, 0.14],
        [112.376, 1738.90, 70.731, 104.945, 147.6, 1522.79, 0.242, 0.08],
        [143.599, 1249.30, 103.652, 54.426, 119.4, 342.36, 0.209, 0.05],
        [105.688, 1337.80, 74.417, 58.843, 220.5, 746.94, 0.223, 0.13]]
data = np.array(data)
# 3 归一化
X, min_val, max_val = critic.min_max_scaler_pos(data)

# 4 运行算法计算权重
W = critic.fit(X)
print(f"权重W = {W}")

# 5计算样本数据评分
test = [117.028, 2532.60, 90.876, 71.531, 315.6, 1301.04, 0.977, 0.17]
X = np.array(test)
X, _, _ = critic.min_max_scaler_pos(X, min_val, max_val)  # 标准化
Z = critic.predict(X)
print(f"预测得分：{Z}")
```
