import numpy as np
import math
import time

class Matrix:
    def __init__(self, A):
        self.A = A

    def __QRDecompose(self, A, standard=False):
        
        m, n = A.shape  
        Q = np.zeros((m, n))
        R = np.zeros((n, n))

        for j in range(n):
            v = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v -= R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            Q[:, j] = v / R[j, j]

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
    
    def __eigenDecompose(self, A):
        # A is a square, symmetric matrix

        n = len(A)
        X = np.copy(A)  # or X = my_copy(A), see below
        pq = np.identity(n)
        max_ct = 10000

        ct = 0
        while ct < max_ct:
            
            Q, R = self.__QRDecompose(X)
            pq = np.matmul(pq, Q)  # accum Q
            X = np.matmul(R, Q)  # note order
            ct += 1

            if self.__is_upper_tri(X, 1.0e-12) == True:
                break

        if ct == max_ct:
            print("WARN: no converge ")

        # eigenvalues are diag elements of X
        e_vals = np.zeros(n, dtype=np.float64)
        for i in range(n):
            e_vals[i] = X[i][i]

        # eigenvectors are columns of pq
        e_vecs = np.copy(pq)
        return (e_vals, e_vecs)
    
    
    def svd(self,B=None):
        if B is None: 
            A = self.A
        else :
            A=B
        ATA = np.matmul(np.transpose(A), A)
        AAT = np.matmul(A, np.transpose(A))

        eigenvals1, eigenvecs1 = self.__eigenDecompose(ATA)
        eigenvals2, eigenvecs2 = self.__eigenDecompose(AAT)

        m,n = A.shape

        diagonal_matrix = np.zeros((m,n))
        z = min(m,n)

        for i in range(z):
            diagonal_matrix[i][i] = math.sqrt(eigenvals1[i])

        for i in range(z):
            Av = np.matmul(A, eigenvecs1[:, i])
            sigmaU = (-1)*diagonal_matrix[i][i]*eigenvecs2[:, i]
            
            result = all(val < 0 for val in Av * sigmaU)
            
            if (result):
                eigenvecs1[:, i] = (-1)*eigenvecs1[:, i]
                

        # print("\nOutputs\n")
        # # print(eigenvals1)
        # print(eigenvecs2)
        # print(diagonal_matrix)
        # print(eigenvecs1)

        return eigenvecs2, diagonal_matrix, eigenvecs1
    
    def randomized_svd(self,k):
        A = self.A
        m, n = A.shape
        p = min(3 * k, n)  # Oversampling parameter

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

    def reduced_svd(self, k):

        U, Sigma, V = self.svd()
        U_reduced = U[:, 0:k]
        Sigma_reduced = Sigma[0:k, 0:k]
        V_reduced = V[:, 0:k]

        return U_reduced, Sigma_reduced, V_reduced      


def main():
    np.set_printoptions(suppress=True, precision=4, floatmode='fixed')

    # Larger example matrix A
    A_large = np.random.rand(100, 80)  # Example of a 100x80 matrix

    print("\nSource matrix: ")
    print(A_large)

    m = Matrix(A_large)

    # Measure time for standard SVD
    start_time = time.time()
    U, Sigma, V = m.svd()
    end_time = time.time()
    print("\nTime taken for standard SVD:", end_time - start_time, "seconds")
    A_prime = np.matmul(U, np.matmul(Sigma, np.transpose(V)))
    print(A_prime)
    # Measure time for reduced SVD
    start_time = time.time()
    U, Sigma, V = m.reduced_svd(20)  # Reduced to 20 dimensions
    end_time = time.time()
    print("Time taken for reduced SVD:", end_time - start_time, "seconds")
    A_prime = np.matmul(U, np.matmul(Sigma, np.transpose(V)))
    print(A_prime)
    # Measure time for randomized SVD
    start_time = time.time()
    U, Sigma, V = m.randomized_svd(20)  # Reduced to 20 dimensions
    end_time = time.time()
    print("Time taken for randomized SVD:", end_time - start_time, "seconds")
    A_prime = np.matmul(U, np.matmul(Sigma, np.transpose(V)))
    print(A_prime)
if __name__ == "__main__":
    main()