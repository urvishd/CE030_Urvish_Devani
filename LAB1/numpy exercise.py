import numpy as np


arr1=np.array([[1,2],[3,4],[5,6]])
arr2=np.array([[1,2,3],[4,5,6]])
arr5=np.array([[4,5],[6,7],[8,9]])

#matrix multiplication
arr3=np.matmul(arr1,arr2)
print(arr3)
print()

#elementwise matrix multiplication
arr4=arr1*arr5
print(arr4)

print()
print('Mean of matrix :',np.mean(arr1))

print()
centered_array=arr2-np.mean(arr2,axis=0)
print(centered_array)