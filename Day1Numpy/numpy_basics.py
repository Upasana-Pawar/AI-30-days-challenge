import numpy as np  

# Create arrays  
arr1 = np.array([1, 2, 3])  
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  

# Indexing  
print(arr2[0, 1])  

# Array math  
print(arr1 + 5)  
print(arr2 * 2)  

# Reshape  
reshaped = arr1.reshape(3, 1)  
print(reshaped)  

# Random numbers  
rand_nums = np.random.randint(1, 10, size=(3, 3))  
print(rand_nums)  
