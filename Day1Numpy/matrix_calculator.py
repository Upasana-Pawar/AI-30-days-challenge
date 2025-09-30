import numpy as np  

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])  

print("Matrix A:\n", A)  
print("Matrix B:\n", B)  

print("Addition:\n", A + B)  
print("Multiplication:\n", np.dot(A, B))  
print("Transpose of A:\n", A.T)  
print("Mean of B:", np.mean(B))  
print("Max of B:", np.max(B))  
print("Min of B:", np.min(B))  
