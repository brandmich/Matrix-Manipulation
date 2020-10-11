import numpy as np
#Created by Brandon Yates for CAP4662, and all work is soley my own work
#compute AB, Ap, A^Tp, A^-1p, p1Xp2, p1^Tp2, A1A2A3A4A5A6A7
#upper case is matrix lower case is vector
#functions check dimension matching when called and will terminate program if invalid dimensions



#Matrix and vector multiplication
print("Matrix/Vector multiplication:")
A = np.array([[3, 6, 7], [5, -3, 0]])
B = np.array([[1, 1], [2, 1], [3, -3]])
p = np.array([[3], [2], [1]])
def matmult(A,B):
    return A.dot(B)

#AB
print("AB:")
C = matmult(A, B)
print(C)
print("\n")
#Ap
print("Ap:")
C = matmult(A, p)
print(C)
print("\n")


#Transpose and multiplication, the first argument being the one that is transposed
print("Transpose then multiplication:")
A2 = np.array([[1, 1], [2, 1], [3, -3]])
p = np.array([[2], [3], [4]])
p2 = np.array([[4], [5], [1]])
def transposemult(A2, p):
    return A2.transpose().dot(p)

#A^Tp
print("A^Tp:")
C = transposemult(A2,p)
print(C)
print("\n")
#p1^Tp2
print("p1^Tp2:")
C = transposemult(p, p2)
print(C)
print("\n")


#works for any inverses, the first argument is the one being inverted
#Inverse then multiplication
print("Inverse multiplication:")
A3 = np.array([[1, 2, 3], [0, 3, -1], [6, 6, 9]])
p = np.array([[1],[5],[1]])
def invMult(A, p):
   return np.linalg.inv(A).dot(p)

#A^-1p
print("A^-1p:")
C = invMult(A3, p)
print(C)
print("\n")

#Cross Product
print("Cross Product:")
p1 = np.array([4, 3])
p2 = np.array([5, 1])
p3 = np.array([4, 3, 1])
p4 = np.array([5, 0, -2])
def cross(p1,p2):
    return np.cross(p1, p2)

#p1Xp2
print("p1Xp2:")
C = cross(p1, p2)
print(C)
print("\n")
print("p3Xp4:")
C = cross(p3, p4)
print(C)
print("\n")
    
#multiply many matrices
print("Multiply Multiple Matrices:")
A = np.array([[3, 6, 7], [5, -3, 0], [3, 5, 4]])
B = np.array([[2, 2, 6], [10, 1, 5], [1, 1, 1]])
C = np.array([[-4, 2, 9], [8, 6, 7], [5, 3, 0]])
D = np.array([[0, 2, 0], [13, -5, 5], [4, 4, 4]])
E = np.array([[2, 2, 4], [6, 1, 9], [5, 3, 2]])
F = np.array([[4, 5, 3], [1, 2, 3], [4, 9, 1]])
G = np.array([[2, 4, 6], [6, 4, 2], [3, 6, 9]])
def manyMat(A, B, C, D, E, F, G):
    temp = A.dot(B)
    temp = temp.dot(C)
    temp = temp.dot(D)
    temp = temp.dot(E)
    temp = temp.dot(F)
    temp = temp.dot(G)
    return temp

#ABCDEFG
print("ABCDEFG:")
mat = manyMat(A, B, C, D, E, F, G)
print(mat)
print("\n")

