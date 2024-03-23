from collections.abc import Iterable
import random
import numpy as np
import scipy.linalg as la
from.Matrix import Matrix

class SparseMatrix:
    def __init__(self, values, nbRows=None, nbColumns=None):
        self.nbRows = nbRows
        self.nbColumns = nbColumns
        self.values = {}

        if nbRows is None or nbColumns is None:
            self.nbRows = len(values)
            self.nbColumns = len(values[0])
            
        if isinstance(values, dict):
            self.values = values.copy()
            
        elif isinstance(values, Iterable):
            if isinstance(values[0], Iterable):
                for i in range(self.nbRows):
                    for j in range(self.nbColumns):
                        if values[i][j] != 0:
                            self.values[(i, j)] = values[i][j]
            else:
                for i in range(self.nbRows):
                    for j in range(self.nbColumns):
                        if values[i*self.nbColumns+j] != 0:
                            self.values[(i, j)] = values[i*self.nbColumns+j]
        else:
            raise TypeError("Invalid values type {}".format(type(values).__name__))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.values.get(index, 0)
        elif isinstance(index, int):
            return self.values.get((index, 0), 0)
        else:
            raise ValueError("Invalid index")

    def __setitem__(self, index, value):
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if value == 0:
                if (i, j) in self.values:
                    del self.values[(i, j)]
            else:
                self.values[(i, j)] = value
        else:
            raise ValueError("Invalid index")

    def __str__(self):
        matrix_str = ""
        for i in range(self.nbRows):
            row_str = ""
            for j in range(self.nbColumns):
                row_str += str(self[i, j]) + " "
            matrix_str += row_str + "\n"
        return matrix_str

    def get_shape(self):
        return (self.nbRows, self.nbColumns)

    def get_values(self):
        return [[self[i, j] for j in range(self.nbColumns)] for i in range(self.nbRows)]

    def transpose(self):
        transposed_values = {(j, i): v for (i, j), v in self.values.items()}
        return SparseMatrix(transposed_values, nbRows=self.nbColumns, nbColumns=self.nbRows)

    def __add__(self, other):
        if isinstance(other, SparseMatrix):
            if self.get_shape() != other.get_shape():
                raise ValueError("Matrices must have the same shape")
            result = [[self[i, j] + other[i, j] for j in range(self.nbColumns)] for i in range(self.nbRows)]
            return SparseMatrix(result, nbRows=self.nbRows, nbColumns=self.nbColumns)
        else:
            raise ValueError("Unsupported operand type(s) for +: 'SparseMatrix' and '{}'".format(type(other).__name__))

    def __sub__(self, other):
        if isinstance(other, SparseMatrix):
            if self.get_shape() != other.get_shape():
                raise ValueError("Matrices must have the same shape")
            result = [[self[i, j] - other[i, j] for j in range(self.nbColumns)] for i in range(self.nbRows)]
            return SparseMatrix(result, nbRows=self.nbRows, nbColumns=self.nbColumns)
        else:
            raise ValueError("Unsupported operand type(s) for -: 'SparseMatrix' and '{}'".format(type(other).__name__))

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            result = {(i, j): v * other for (i, j), v in self.values.items()}
            return SparseMatrix(result, nbRows=self.nbRows, nbColumns=self.nbColumns)
        elif isinstance(other, SparseMatrix):
            if self.nbColumns != other.nbRows:
                raise ValueError("The number of columns of the first matrix must be equal to the number of rows of the second matrix")
            result = {}
            for i in range(self.nbRows):
                for j in range(other.nbColumns):
                    result[(i, j)] = sum(self[i, k] * other[k, j] for k in range(self.nbColumns))
            return SparseMatrix(result, nbRows=self.nbRows, nbColumns=other.nbColumns)
        else:
            raise ValueError("Unsupported operand type(s) for *: 'SparseMatrix' and '{}'".format(type(other).__name__))

    def __eq__(self, other):
        if not isinstance(other, SparseMatrix):
            return False
        if self.get_shape() != other.get_shape():
            return False
        for i in range(self.nbRows):
            for j in range(self.nbColumns):
                if self[i, j] != other[i, j]:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
    
    import random

class SparseMatrix:
    # Existing methods from previous implementation

    def getDeterminant(self):
        if self.nbRows != self.nbColumns:
            raise ValueError("The matrix must be square")
        elif self.nbRows == 1:
            return self[0, 0]
        elif self.nbRows == 2:
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        else:
            # Implement determinant calculation for larger matrices if needed
            raise NotImplementedError("Determinant calculation for larger matrices is not implemented")

    def getMinor(self, i, j):
        submatrix_values = {(r, c): v for (r, c), v in self.values.items() if r != i and c != j}
        return SparseMatrix(submatrix_values, nbRows=self.nbRows - 1, nbColumns=self.nbColumns - 1)

    def inverse(self):
        if self.nbRows != self.nbColumns:
            raise ValueError("The matrix must be square")
        determinant = self.getDeterminant()
        if determinant == 0:
            raise ValueError("The matrix is not invertible")
        cofactors = {(i, j): (-1) ** (i + j) * self.getMinor(i, j).getDeterminant() for i in range(self.nbRows) for j in range(self.nbColumns)}
        adjugate_values = {(j, i): cofactors[(i, j)] / determinant for i in range(self.nbRows) for j in range(self.nbColumns)}
        return SparseMatrix(adjugate_values, nbRows=self.nbRows, nbColumns=self.nbColumns)

    def solve_linear_system(self, b):
        if self.nbRows != self.nbColumns:
            raise ValueError("The matrix must be square")
        if len(b) != self.nbRows:
            raise ValueError("The number of rows of the matrix must be equal to the length of the vector b")
        augmented_matrix_values = self.values.copy()
        for i in range(self.nbRows):
            augmented_matrix_values[(i, self.nbColumns)] = b[i]
        augmented_matrix = SparseMatrix(augmented_matrix_values, nbRows=self.nbRows, nbColumns=self.nbColumns + 1)
        # Perform Gaussian elimination on augmented matrix
        for i in range(len(augmented_matrix)-1):
            pivot_row = max(range(i, len(augmented_matrix)), key=lambda j: abs(augmented_matrix[j][i]))
            augmented_matrix[i], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[i]

            pivot = augmented_matrix[i, i]
            for j in range(i+1, len(augmented_matrix)):
                factor = augmented_matrix[j, i] / pivot
                for k in range(i, len(augmented_matrix[0])):
                    augmented_matrix[j, k] -= factor * augmented_matrix[i][k]

        # Back Substitution
        x = [0] * len(self)
        for i in range(len(augmented_matrix)-1, -1, -1):
            x[i] = augmented_matrix[i, -1]
            for j in range(i+1, len(augmented_matrix[0])-1):
                x[i] -= augmented_matrix[i, j] * x[j]
            x[i] /= augmented_matrix[i, i]

        return x

    def threshold_matrix(self, threshold):
        thresholded_values = {(r, c): v if abs(v) >= threshold else 0 for (r, c), v in self.values.items()}
        return SparseMatrix(thresholded_values, nbRows=self.nbRows, nbColumns=self.nbColumns)

    def toInt(self):
        integer_values = {(r, c): int(v) for (r, c), v in self.values.items()}
        return SparseMatrix(integer_values, nbRows=self.nbRows, nbColumns=self.nbColumns)

    def copy(self):
        return SparseMatrix(self.values.copy(), nbRows=self.nbRows, nbColumns=self.nbColumns)

    def randomize(self, low=0, high=1):
        random_values = {(r, c): random.uniform(low, high) for r in range(self.nbRows) for c in range(self.nbColumns)}
        return SparseMatrix(random_values, nbRows=self.nbRows, nbColumns=self.nbColumns)

    def randint(self, low, high):
        random_values = {(r, c): random.randint(low, high) for r in range(self.nbRows) for c in range(self.nbColumns)}
        return SparseMatrix(random_values, nbRows=self.nbRows, nbColumns=self.nbColumns)

    def randomFloatMatrix(self, low, high):
        random_values = {(r, c): random.uniform(low, high) for r in range(self.nbRows) for c in range(self.nbColumns)}
        return SparseMatrix(random_values, nbRows=self.nbRows, nbColumns=self.nbColumns)

    def scipyEigenvalues(self):
        # Convert SparseMatrix to dense matrix and then use scipy.linalg.eigvals
        dense_matrix = self.to_dense()
        return la.eigvals(dense_matrix)

    def numpyEigenvalues(self):
        # Convert SparseMatrix to dense matrix and then use numpy.linalg.eigvals
        dense_matrix = self.to_dense()
        return np.linalg.eigvals(dense_matrix)
    
    def norm(self):
        squared_sum = sum(v ** 2 for v in self.values.values())
        return squared_sum ** 0.5

    def dot_product(self, other):
        if self.nbColumns != other.nbRows:
            raise ValueError("Number of columns of the first matrix must be equal to the number of rows of the second matrix")
        result = sum(self[i, j] * other[j, k] for i in range(self.nbRows) for j in range(self.nbColumns) for k in range(other.nbColumns))
        return result

    def trace(self):
        if self.nbRows != self.nbColumns:
            raise ValueError("The matrix must be square")
        return sum(self[i, i] for i in range(self.nbRows))

    # Method for matrix-vector multiplication
    def matvec(self, vector):
        if len(vector) != self.nbColumns:
            raise ValueError("The length of the vector must be equal to the number of columns of the matrix")
        result = [sum(self[i, j] * vector[j] for j in range(self.nbColumns)) for i in range(self.nbRows)]
        return result

    # Method for vector-matrix multiplication
    def vecmat(vector, matrix):
        if len(vector) != matrix.nbRows:
            raise ValueError("The length of the vector must be equal to the number of rows of the matrix")
        result = [sum(vector[i] * matrix[i, j] for i in range(matrix.nbRows)) for j in range(matrix.nbColumns)]
        return result

    def to_dense(self):
        dense_matrix = [[self[i, j] for j in range(self.nbColumns)] for i in range(self.nbRows)]
        return dense_matrix
    
    def toArray(self):
        matrix_array = [[self[i, j] for j in range(self.nbColumns)] for i in range(self.nbRows)]
        return np.array(matrix_array)
    
    def toMatrix(self):
        return Matrix(self.toArray())
