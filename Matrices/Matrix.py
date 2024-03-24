import numpy as np
import scipy.linalg as la
import random
from collections.abc import Iterable

def importVector():
    from .Vector import Vector
    return Vector

def importFunctions():
    import Functions as f
    return f
    
class Matrix:
    
    def __init__(self, values : list[list], *, mutable = False, nbColumns = None, nbRows = None) -> None:

        if isinstance(values[0], Iterable):
            if (nbColumns, nbRows) not in ((len(values[0]), len(values)), (None, None)):
                raise ValueError("Got unmatching nbColumns ({}) and nbRows ({}) for a {}x{} matrix".format(nbColumns, nbRows, len(values[0]), len(values)))

            self.values             = list(sum(values, type(values[0])())) if mutable else tuple(sum(values, type(values[0])())) 

            self._nbColumns : int   = len(values[0])
            self._nbRows : int      = len(values)

        else:
            
            if (nbColumns, nbRows) == (None, None):
                raise Exception("nbColumns and nbRows unspecified")
            
            if nbColumns != None:
                if len(values) % nbColumns:
                    raise Exception("Impossible value for nbColumns (got: {}, remider of len(values)//nbColumns: {})".format(nbColumns, len(values) % nbColumns))
                
                self._nbColumns = nbColumns
                self._nbRows = len(values) // nbColumns
            
            else:
                if len(values) % nbRows:
                    raise Exception("Impossible value for nbRows (got: {}, remider of len(values)//nbColumns: {})".format(nbRows, len(values) % nbRows))
                
                self._nbColumns = len(values) // nbRows
                self._nbRows = nbRows
                
            self.values = list(values) if mutable else tuple(values)
        
        self.__mutable : bool   = mutable
    
    def get_values(self) -> tuple:
        return tuple((self.values[i*self._nbColumns:(i+1)*self._nbColumns] for i in range(self._nbRows)))
    
    def isMutable(self) -> bool:
        return self.__mutable
    
    def get_shape(self) -> tuple:
        return (self._nbRows, self._nbColumns)
    
    def getNbCol(self) -> int:
        return self._nbColumns
    
    def getNbRows(self) -> int:
        return self._nbRows
    
    def getDeterminant(self) -> float:
        if self._nbColumns != self._nbRows:
            raise ValueError("The matrix must be square")
        elif self._nbColumns == 1:
            return self.values[0]
        elif self._nbColumns == 2:
            return self.values[0]*self.values[3] - self.values[1]*self.values[2]
        return sum(self.values[i] * self.getMinor(0, i).getDeterminant() * (-1)**i for i in range(self._nbColumns))
    
    def getMinor(self, i : int, j : int) -> 'Matrix':
        return Matrix([self.values[k] for k in range(len(self.values)) if k//self._nbColumns != i and k%self._nbColumns != j], nbColumns=self._nbColumns-1)

    def transpose(self) -> 'Matrix':
        return Matrix([[self.values[j*self._nbColumns+i] for j in range(self._nbRows)] for i in range(self._nbColumns)])
    
    def inverse(self) -> 'Matrix':
        if self._nbColumns != self._nbRows:
            raise ValueError("The matrix must be square")
        determinant = self.getDeterminant()
        if determinant == 0:
            raise ValueError("The matrix is not invertible")
        return Matrix([[self.getMinor(j, i).getDeterminant() * (-1)**(i+j) / determinant for j in range(self._nbColumns)] for i in range(self._nbRows)])
    
    def solve_linear_system(self, b):
        Vector = importVector()  # Avoiding circular import
        
        # Construct augmented matrix [A | b]
        augmented_matrix = Matrix([row + [bi] for row, bi in zip(self.get_values(), b.get_values())], mutable=True)

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
        x = Vector([0] * len(self[0]), mutable=True)
        for i in range(len(augmented_matrix)-1, -1, -1):
            x[i] = augmented_matrix[i, -1]
            for j in range(i+1, len(augmented_matrix[0])-1):
                x[i] -= augmented_matrix[i, j] * x[j]
            x[i] /= augmented_matrix[i, i]

        return x
    
    def threshold_matrix(self, threshold):
        # return Matrix([map(lambda x : 1 if x >= threshold else 0, row) for row in self.get_values()]) # with a Lamda function
        return Matrix([[1 if element >= threshold else 0 for element in row] for row in self.get_values()])
    
    def toInt(self):
        return Matrix([int(v) for v in self.values], nbColumns=self._nbColumns, mutable=self.__mutable)
    
    #--- Using external libraries ---
        #--- Numpy / Scipy---
    def toArray(self) -> np.ndarray:
        return np.array(self.get_values())
    
    def scipyEigenvalues(self) -> np.ndarray:
        return la.eigvals(self.toArray())
    
    def numpyEigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.toArray())
    
        #--- Random ---
    @staticmethod
    def randomMatrix(nbRows : int, nbColumns : int, *, mutable = False) -> 'Matrix':
        return Matrix([[random.random() for _ in range(nbColumns)] for _ in range(nbRows)], mutable=mutable)
    
    @staticmethod
    def randintMatrix(nbRows : int, nbColumns : int, low : int, high : int, *, mutable = False) -> 'Matrix':
        return Matrix([[random.randint(low, high) for _ in range(nbColumns)] for _ in range(nbRows)], mutable=mutable)
    
    @staticmethod
    def randomFloatMatrix(nbRows : int, nbColumns : int, low : float, high : float, *, mutable = False) -> 'Matrix':
        return Matrix([[random.uniform(low, high) for _ in range(nbColumns)] for _ in range(nbRows)], mutable=mutable)
    
    #--- Basic operations ---
    def copy(self) -> 'Matrix':
        return Matrix(self.get_values())
    
    def __add__(self, other) -> 'Matrix':
        Vector = importVector()  # Avoiding circular import
        
        if isinstance(other, Vector):
            if other.isColVector:
                if self._nbColumns != other._nbColumns:
                    raise ValueError("The number of columns of the matrix must be equal to the number of elements of the vector")
                return Matrix([[self.values[i*self._nbColumns+j] + other.values[j] for j in range(self._nbColumns)] for i in range(self._nbRows)])
            else:
                if self._nbRows != other._nbRows:
                    raise ValueError("The number of rows of the matrix must be equal to the number of elements of the vector")
                return Matrix([[self.values[i*self._nbColumns+j] + other.values[i] for j in range(self._nbColumns)] for i in range(self._nbRows)])
        
        if self.get_shape() != other.get_shape():
            raise ValueError("Matrices must have the same shape")
        
        return Matrix([[self.values[i*self._nbColumns+j] + other.values[i*other.nbColumns+j] for j in range(self._nbColumns)] for i in range(self._nbRows)])
    
    def __iadd__(self, other) -> 'Matrix':
        return self + other
    
    def __sub__(self, other) -> 'Matrix':
        return self + other*(-1)
    
    def __isub__(self, other) -> 'Matrix':
        return self - other

    def __mul__(self, other) -> 'Matrix':
        Vector = importVector()  # Avoiding circular import
        
        if type(other) in (int, float):
            return Matrix([[self.values[i*self._nbColumns+j] * other for j in range(self._nbColumns)] for i in range(self._nbRows)])
        
        if self._nbColumns != other._nbRows:
            raise ValueError("The number of columns of the first matrix must be equal to the number of rows of the second matrix")
        
        values = [[sum(self.values[i*self._nbColumns+k] * other.values[k*other._nbColumns+j] for k in range(self._nbColumns)) for j in range(other._nbColumns)] for i in range(self._nbRows)]
        
        if isinstance(other, Vector):
            return Vector(values)
            
        return Matrix(values)
        
    
    def __imul__(self, other) -> 'Matrix':
        return self * other
    
    def __eq__(self, other) -> bool:
        return self.values == other.values
    
    def __str__(self) -> str:
        return str("\n".join((str(v) for v in self.get_values())))
    
    def __getitem__(self, index):
        if isinstance(index, Iterable):
            if len(index) != 2:
                raise ValueError("Invalid index")
            i, j = index
            return self.get_values()[i][j]
        else:
            return self.get_values()[index]
        
    def get_column(self, j):
        Vector = importVector()
        return Vector([self.data[i][j] for i in range(self._nbRows)])
    
    def get_row(self, i):
        Vector = importVector()
        return Vector(self.data[i])
    
    def __setitem__(self, index, value):
        
        if not(self.isMutable()):
            raise ValueError("The matrix is not mutable")
        
        if not(isinstance(index, Iterable)):
            index = tuple([index, None])
        
        if len(index) != 2:
            raise ValueError("Invalid index")
        
        i, j = index
        if j == -1:  # Add entire row
            self.add_row(value)
        elif i == -1:  # Add entire column
            self.add_column(value)
        elif j is None:  # Assign entire row
            self.set_row(i, value)
        elif i is None:  # Assign entire column
            self.set_column(j, value)
        else:
            self.values[i * self._nbColumns + j] = value
        
    def add_row(self, values):
        if len(values) != self._nbColumns:
            raise ValueError("Number of values must match the number of columns")
        self._nbRows += 1
        self.values.extend(values)

    def add_column(self, values):
        if len(values) != self._nbRows:
            raise ValueError("Number of values must match the number of rows")
        for i in range(self._nbRows):
            self.values.insert((i + 1) * self._nbColumns + i, values[i])
        self._nbColumns += 1

    def set_row(self, index, values):
        if len(values) != self._nbColumns:
            raise ValueError("Number of values must match the number of columns")
        if not (0 <= index < self._nbRows):
            raise ValueError("Row index out of range")
        self.values[index * self._nbColumns: (index + 1) * self._nbColumns] = values

    def set_column(self, index, values):
        if len(values) != self._nbRows:
            raise ValueError("Number of values must match the number of rows")
        if not (0 <= index < self._nbColumns):
            raise ValueError("Column index out of range")
        for i in range(self._nbRows):
            self.values[i * self._nbColumns + index] = values[i]
    
    def __len__(self) -> int:
        return self._nbRows
    
    def __iter__(self):
        return iter(self.get_values())
    
    def get_column(self, j):
        Vector = importVector()
        return Vector([self.data[i][j] for i in range(self._nbRows)])
    
    @staticmethod
    def create_diagonal(values : list, *, mutable = False) -> 'Matrix':
        return Matrix([[values[i] if i == j else 0 for j in range(len(values))] for i in range(len(values))], mutable=mutable)
    
    def set_column(self, j, col):
        for i in range(self._nbRows):
            self.data[i][j] = col.elements[i]

    def normalize_columns(self):
        for j in range(self._nbColumns):
            col = self.get_column(j)
            col.normalize()
            self.set_column(j, col)
            
    def matrix_norm(self, norm_type: str = 'L2') -> float:
        """
        Compute the L1, L2, or Linf norm of a given matrix.

        Parameters:
            matrix (list[list[int]]): The input matrix.
            norm_type (str): The type of norm to compute. Options: 'L1', 'L2', or 'Linf'.

        Returns:
            float: The computed norm.

        Raises:
            ValueError: If an invalid norm type is specified.
        """
        n_rows, n_cols = self.get_shape()
        if norm_type == 'L1':
            # max of the sum of the absolute values for each columns
            return max([sum([abs(e) for e in self.get_column(i)]) for i in range(n_cols)])
        elif norm_type == 'L2':
            f = importFunctions()
            #square root of the max eigenvalue
            return max(f.qr_algorithm(self)[0])**0.5
        elif norm_type == 'Linf':
            # max of the sum of the absolute values for each rows
            return max([sum([abs(e) for e in self.get_row(i)]) for i in range(n_rows)])
        else:
            raise ValueError("Invalid norm type. Please choose from 'L1', 'L2', or 'Linf'.")
    
    def __iter__(self):
        nb_rows, nb_columns = self.get_shape()
        for i in range(nb_rows*nb_columns):
            yield self.values[i]