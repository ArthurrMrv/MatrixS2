from .Matrix import Matrix
from .Vector import Vector

def qr_algorithm(A: Matrix, num_iterations: int = 10) -> tuple[list, list]:
    """Find the eigenvalues and eigenvectors of a matrix using the QR algorithm.

    Args:
        A (Matrix): matrix to find the eigenvalues and eigenvectors of.
        num_iterations (int, optional): number of iterations . Defaults to 10.

    Returns:
        tuple[list, list]: eigenvalues and eigenvectors of the matrix.
    """
    Q = Matrix([[0] * A._nbRows for _ in range(A._nbRows)], mutable=True)
    R = Matrix([[0] * A._nbColumns for _ in range(A._nbColumns)], mutable=True)
    
    for _ in range(num_iterations):
        # Perform QR decomposition
        for j in range(A._nbColumns):
            v = Vector([[e] for e in A[j]])
            for i in range(j):
                vect_Qi = Vector(Q[i])
                R[i, j] = Vector.dotProduct(vect_Qi, v)
                v = v - (vect_Qi * R[i, j])

            norm_v = v.norm()
            if norm_v == 0:  # Handle division by zero
                continue

            R[j, j] = norm_v
            Q[j] = [x[0] / R[j, j] for x in v]

        # Update A with RQ
        A = R * Q

    eigenvalues = [A[i][i] for i in range(A._nbRows)]
    eigenvectors = [Vector(Q[i]) for i in range(Q._nbRows)]
    
    return eigenvalues, eigenvectors

def qr_decomposition(A : Matrix) -> tuple[Matrix, Matrix]:

    Q = Matrix([[0] * A._nbRows for _ in range(A._nbRows)], mutable=True)
    R = Matrix([[0] * A._nbColumns for _ in range(A._nbColumns)], mutable=True)

    for j in range(A._nbColumns):
        v = Vector([[e] for e in A[j]])
        for i in range(j):
            vect_Qi = Vector(Q[i])
            R[i, j] = Vector.dotProduct(vect_Qi, v)
            v = v - (vect_Qi * R[i, j])

        norm_v = v.norm()
        if norm_v == 0:  # Handle division by zero
            continue

        R[j, j] = norm_v
        #R[j][j] = v.norm()
        Q[j] = [x[0] / R[j, j] for x in v]

    return Q, R

def getSVD_qr(A : Matrix) -> tuple[Matrix, list, Matrix]:
    Q, R    = qr_decomposition(A)
    U       = Q.copy()                              # Left singular vectors
    Sigma   = [R[i, i] for i in range(R._nbRows)]   # Singular values
    VT      = Q.transpose() * A                     # Right singular vectors
    return U, Sigma, VT.transpose()

def solve_linear_system(A : Matrix, b : Vector) -> Vector:    
    # Construct augmented matrix [A | b]
    augmented_matrix = Matrix([row + [bi] for row, bi in zip(A.get_values(), b.get_values())], mutable=True)

    # Gaussian Elimination
    for i in range(len(augmented_matrix)-1):
        pivot_row = max(range(i, len(augmented_matrix)), key=lambda j: abs(augmented_matrix[j][i]))
        augmented_matrix[i], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[i]
        
        pivot = augmented_matrix[i, i]
        for j in range(i+1, len(augmented_matrix)):
            factor = augmented_matrix[j, i] / pivot
            for k in range(i, len(augmented_matrix[0])):
                augmented_matrix[j, k] -= factor * augmented_matrix[i][k]

    # Back Substitution
    x = Vector([0] * len(A[0]), mutable=True)
    for i in range(len(augmented_matrix)-1, -1, -1):
        x[i] = augmented_matrix[i, -1]
        for j in range(i+1, len(augmented_matrix[0])-1):
            x[i] -= augmented_matrix[i, j] * x[j]
        x[i] /= augmented_matrix[i, i]

    return x