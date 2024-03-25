import time
from Matrices import Matrix
import cProfile

def main():
    cProfile.run("run_tests()", "timesMatrices.prof")
    
def get_time(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

def run_tests():
    matrix = Matrix.randomMatrix(10, 10)
    
    print("Computation time for each method of a 10x10 matrix:")
    print("-----------------------------------------------------")
    
    print("getDeterminant:", get_time(matrix.getDeterminant))
    print("getMinor:", get_time(matrix.getMinor, 0, 0))
    print("transpose:", get_time(matrix.transpose))
    print("inverse:", get_time(matrix.inverse))
    print("threshold_matrix:", get_time(matrix.threshold_matrix, 0.5))
    print("toInt:", get_time(matrix.toInt))
    print("toArray:", get_time(matrix.toArray))
    print("scipyEigenvalues:", get_time(matrix.scipyEigenvalues))
    print("numpyEigenvalues:", get_time(matrix.numpyEigenvalues))
    print("randomMatrix:", get_time(Matrix.randomMatrix, 100, 100))
    print("randintMatrix:", get_time(Matrix.randintMatrix, 100, 100, 0, 10))
    print("randomFloatMatrix:", get_time(Matrix.randomFloatMatrix, 100, 100, 0.0, 1.0))
    print("copy:", get_time(matrix.copy))
    print("__add__:", get_time(lambda: matrix + matrix))
    print("__sub__:", get_time(lambda: matrix - matrix))
    print("__mul__:", get_time(lambda: matrix * matrix))
    print("__eq__:", get_time(lambda: matrix == matrix))
    print("get_column:", get_time(matrix.get_column, 0))
    print("get_row:", get_time(matrix.get_row, 0))
    
    # For mutable Matrices only
    #print("__setitem__:", get_time(lambda: matrix[0][0] = 1))
    #print("add_row:", get_time(lambda: matrix.add_row([1]*100)))
    #print("add_column:", get_time(lambda: matrix.add_column([1]*100)))
    #print("set_row:", get_time(lambda: matrix.set_row(0, [1]*100)))
    #print("set_column:", get_time(lambda: matrix.set_column(0, [1]*100)))
    
    print("matrix_norm:", get_time(matrix.matrix_norm, 'L2'))

if __name__ == "__main__":
    main()
