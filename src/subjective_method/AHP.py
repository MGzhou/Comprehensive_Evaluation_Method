
import numpy as np


class AHP:
    #传入的np.ndarray是的判断矩阵
    def __init__(self,array):
        self.array = array
        # 记录矩阵大小
        self.n = array.shape[0]
        # 初始化RI值,用于一致性检验 
        # RI_list = [0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45] # 这个表来源不知道
        RI_list = [0,0,0.52,0.89,1.12,1.26,1.36,1.41,1.46]
        self.RI = RI_list[self.n-1]
        if self.n > 9:
            CR = None
            print(f"由于RI表只有9阶,因此无法进行一致性校验，请扩充RI表")
        
	#获取最大特征值和对应的特征向量
    def get_eig(self):
        #numpy.linalg.eig() 计算矩阵特征值与特征向量
        eig_val ,eig_vector = np.linalg.eig(self.array)
        #获取最大特征值
        max_val = np.max(eig_val)
        max_val = round(max_val.real, 4)
        #通过位置来确定最大特征值对应的特征向量
        index = np.argmax(eig_val)
        max_vector = eig_vector[:,index]
        max_vector = max_vector.real.round(4) 
        #添加最大特征值属性
        self.max_val = max_val
        #计算权重向量W
        weight_vector = max_vector/sum(max_vector)
        weight_vector = weight_vector.round(4)
        #打印结果
        print("最大的特征值: "+str(max_val))
        print("对应的特征向量为: "+str(max_vector))
        print("归一化后得到权重向量: "+str(weight_vector))
        return weight_vector
    
    #测试一致性
    def test_consitst(self):
        #计算CI值
        CI = (self.max_val-self.n)/(self.n-1) 
        CI = round(CI,4) 
        #打印结果
        print("判断矩阵的CI值为" +str(CI))
        print("判断矩阵的RI值为" +str(self.RI))
        #分类讨论
        if self.n == 2:
            print("仅包含两个子因素，不存在一致性问题")
        else:
            #计算CR值
            CR = CI/self.RI 
            CR = round(CR,4)
            #CR < 0.10才能通过检验
            if  CR < 0.10 :
                print("判断矩阵的CR值为" +str(CR) + "，通过一致性检验")
                return True
            else:
                print("判断矩阵的CR值为" +str(CR) + "，未通过一致性检验")
                return False

A = [
    [1, 1./3, 1./2, 1./2],
    [3, 1, 2, 2],
    [2, 1./2, 1, 2],
    [2, 1/2, 1/2, 1]
]
A = np.array(A)

ahp1 = AHP(A)
w = ahp1.get_eig()
print(f"权重= {w}")
# 一致性
ahp1.test_consitst()