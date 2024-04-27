import numpy as np

  
def min_max_scaler_2d(data):  
    min_val = np.min(data, axis=0)  
    max_val = np.max(data, axis=0)  
    return (max_val-data) / (max_val - min_val)  


# 假设我们有一个二维numpy数组  
data = np.array([[1, 2, 3],  
                  [4, 5, 6],  
                  [7, 8, 9]])  

# 使用min_max_scaler_2d函数对每一列进行标准化  
normalized_data = min_max_scaler_2d(data)  
  
print(normalized_data)