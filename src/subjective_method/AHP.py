
import numpy as np


class AHP:
    def __init__(self, judgment_matrix):
        """
        @judgment_matrix: 传入的np.ndarray是的判断矩阵
        """
        self.judgment_matrix = judgment_matrix
        self.n = judgment_matrix.shape[0] # 记录矩阵大小
        self.max_val = None # 最大特征根
        self.max_vector = None # 特征向量
        self.weight_vector = None # 权重（最终需要的结果）
        self.CI = None
        self.CR = None
        # 初始化RI值,用于一致性检验 
        RI_list = [
            0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 
            1.49, 1.52, 1.54, 1.56, 1.58, 1.59, 1.5943, 1.6064,
            1.6133, 1.6207, 1.6292
        ]
        self.RI = RI_list[self.n-1]
        if self.n > 20:
            raise "由于RI表只有20阶,因此无法进行一致性校验，请扩充RI表"

    def fit(self):
        """
        计算权重
        """
        #numpy.linalg.eig() 计算矩阵特征值与特征向量
        eig_val ,eig_vector = np.linalg.eig(self.judgment_matrix)
        #获取最大特征值
        max_val = np.max(eig_val)
        #通过位置来确定最大特征值对应的特征向量
        index = np.argmax(eig_val)
        max_vector = eig_vector[:,index]
        self.max_vector = max_vector.real
        #添加最大特征值属性
        self.max_val = max_val
        #计算权重向量W
        self.weight_vector = self.max_vector/sum(self.max_vector)
        #打印结果
        return self.weight_vector
    
    def test_consitst(self):
        """
        一致性校验.
        注意: 需要先计算权重, 即先运行fit()函数.
        """
        #计算CI值
        CI = (self.max_val-self.n)/(self.n-1) 
        self.CI = round(CI,4)
        #分类讨论
        if self.n <= 2:
            print("仅包含两个子因素，不存在一致性问题")
        else:
            #计算CR值
            CR = CI/self.RI 
            self.CR = round(CR,4)
            #CR < 0.10才能通过检验
            if  CR < 0.10 :
                print("判断矩阵的CR值为" +str(self.CR) + "，通过一致性检验")
                return True
            else:
                print("判断矩阵的CR值为" +str(self.CR) + "，未通过一致性检验")
                return False
    
    def predict(self, X):
        """
        @X: 数据样本，类型需要 np.ndarray
        """
        return X @ self.weight_vector


if __name__=="__main__":
    A = [
        [1, 1./3, 1./2, 1./2],
        [3, 1, 2, 2],
        [2, 1./2, 1, 2],
        [2, 1/2, 1/2, 1]
    ]
    A = np.array(A)
    ahp1 = AHP(A)
    # 模型计算
    ahp1.fit()
    # 一致性
    ahp1.test_consitst()

    #打印结果
    print("判断矩阵的CI值为" +str(ahp1.CI))
    print("判断矩阵的RI值为" +str(ahp1.RI))
    print("最大的特征值: "+str(ahp1.max_val))
    print("对应的特征向量为: "+str(ahp1.max_vector))
    print("归一化后得到权重向量: "+str(ahp1.weight_vector))
    print(f"权重相加：{sum(ahp1.weight_vector)}")

    # 预测
    X = [[90, 20, 50, 30]]
    target = ahp1.predict(X)
    print(f"样本得分= {target.round(4)}")

    