import numpy as np
import math

class Matrix:
    def __init__(self, A):
        self.A = A

    def __QRDecompose(self, A, standard=False,flg = True):
        
        m, n = A.shape  
        Q = np.zeros((m, n))
        R = np.zeros((n, n))

    
        # for j in range(n):
        #     v = np.copy(A[:, j])
        #     if j > 0:
        #         R[:j, j] = v.T @ Q[:,:j]  # Matrix multiplication for all previous columns
        #         v -= Q[:, :j] @ R[:j, j]         # Subtract projections of previous columns
               
        #     R[j, j] = np.linalg.norm(v)
        #     # print(R[j,j])
        #     Q[:, j] = v / (R[j, j] + 1e-9)

        for j in range(n):
            v = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v -= R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            Q[:, j] = v / (R[j, j] + 1e-9)

        if standard:
            for i in range(n):
                if R[i, i] < 0.0:
                    Q[:, i] *= -1
                    R[i, :] *= -1

        return Q, R
    
    def __is_upper_tri(self, A, tol):
        n = len(A)
        for i in range(0,n):
            for j in range(0,i):
            # if the lower triangular entries fall below a threshold (tol), only then it is considered an upper triangular matrix, else not
                if np.abs(A[i][j]) > tol:
                    return False
        return True
    
    def eigenDecomposeSelf(self):
        return self.__eigenDecompose(self.A)
    
    def __eigenDecompose(self, A,flg):
        # A is a square, symmetric matrix

        n = len(A)
        X = np.copy(A)  # or X = my_copy(A), see below
        pq = np.identity(n)
        max_ct =1000

        ct = 0
        while ct < max_ct:
            
            Q, R = self.__QRDecompose(X,flg)
           
            pq = np.matmul(pq, Q)  # accum Q
            X = np.matmul(R, Q)  # note order
            ct += 1
            

            if self.__is_upper_tri(X, 1.0e-9) == True:
                
                break

        if ct == max_ct:
            print("WARN: no converge ")

        # eigenvalues are diag elements of X
        e_vals = np.zeros(n, dtype=np.float64)
        for i in range(n):
            e_vals[i] = X[i][i]

        # eigenvectors are columns of pq
        e_vecs = np.copy(pq)
        print("These are evecs")
        print(e_vecs)
        print(e_vals)
        return (e_vals, e_vecs)
    
  
    def svd(self,B=None,flg = True):
        if B is None: 
            A = self.A
        else :
            A=B
        # A=self.A

        ATA = np.matmul(np.transpose(A), A)
        # AAT = np.matmul(A, np.transpose(A))
        # print("ATA")
        # print(ATA)
        eigenvals1, eigenvecs1 = self.__eigenDecompose(ATA,flg)
        eigenvals1[eigenvals1 < 1e-9] = 0
        # print("Eigenvals")
        # print(eigenvals1)
        # eigenvals2, eigenvecs2 = self.__eigenDecompose(AAT)

        ## need to find way to do only once

        m,n = A.shape

        diagonal_matrix = np.zeros((m,n))
        z = min(m,n)

        eigenvecs2 = np.zeros((m,m))
        # print(eigenvecs2)

        for i in range(z):
            # print(eigenvals1[i])
            diagonal_matrix[i][i] = math.sqrt(eigenvals1[i])

        for i in range(z):
            Av = np.matmul(A, eigenvecs1[:, i])
            u = Av/(diagonal_matrix[i][i]+ 1e-9)
            # print(u.shape)
            eigenvecs2[:, i] = u

            # sigmaU = (-1)*diagonal_matrix[i][i]*eigenvecs2[:, i]
            
            # result = all(val < 0 for val in Av * sigmaU)
            # result = all(val < 0 for val in Av * sigmaU)
            
            # if (result):
            #     eigenvecs1[:, i] = (-1)*eigenvecs1[:, i]
            # if (result):
            #     eigenvecs1[:, i] = (-1)*eigenvecs1[:, i]
                

        # print("\nOutputs\n")
        # print(eigenvecs2)
        # print(diagonal_matrix)
        # print(eigenvecs1)

        return eigenvecs2, diagonal_matrix, eigenvecs1

    
    def reduced_svd(self, k):

        U, Sigma, V = self.svd()
        U_reduced = U[:, 0:k]
        Sigma_reduced = Sigma[0:k, 0:k]
        V_reduced = V[:, 0:k]

        return U_reduced, Sigma_reduced, V_reduced      

    def randomized_svd(self,k):
        A = self.A
        m, n = A.shape
        p = min(2*k, n)  # Oversampling parameter

        # Generate a random Gaussian matrix
        Omega = np.random.randn(n, p)
        Y = np.dot(A, Omega)
        Q, _ = self.__QRDecompose(Y)
        B = np.dot(Q.T, A)
        U_hat, Sigma, V = self.svd(B)
        U = np.dot(Q, U_hat)

        # Truncate to the top k singular values/vectors
        U = U[:, 0:k]
        Sigma = Sigma[0:k,0:k]
        V = V[:, :k]
        return U, Sigma, V
    
    
def main():
    np.set_printoptions(suppress=True,
    precision=4, floatmode='fixed')

    A = np.array([[0.9, 0.8,0.3],[0.6,0.8,0.7], [0.2 ,0.3,0.3],[0.6,0.8,0.7],[0.3, 0.1, 0.3],[0.6,0.8,0.7]], dtype=np.float64)

    print("\nSource matrix: ")
    print(A)

    m = Matrix(A)

    U, Sigma, V = m.svd()
    A_prime = np.matmul(U, np.matmul(Sigma, np.transpose(V)))

    print("\nAfter SVD")
    print(A_prime)

    
    U, Sigma, V = m.randomized_svd(2)
    A_prime = np.matmul(U, np.matmul(Sigma, np.transpose(V)))
    print("\nReduced SVD")
    print(A_prime)


if __name__ == "__main__":
    main()