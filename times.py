import time
from Matrices import Matrix
import cProfile
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    #cProfile.run("run_tests()", "data/timesMatrices.cprof")
    #run_tests()
    save_all_resuts([i for i in range(10, 1_000, 10)])
    
def get_time(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

def run_tests():
    matrix = Matrix.randomMatrix(20, 20)
    
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
    
def save_all_resuts(size_list: list):
    results = dict()
    for i in tqdm(range(len(size_list)), desc="Testing matrix sizes"):
        plt.figure()
        temp = get_test_results(size_list[i])
        for k in temp.keys():
            if not(k in results):
                results[k] = [temp[k]]
            else:
                results[k].append(temp[k])
        for k in results.keys():
            plt.plot(size_list[:(i+1)], results[k], label=k)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("Matrix size")
        plt.ylabel("Time (s)")
        plt.title("Execution time of each method for different matrix sizes")
        # Adjust figure size to accommodate legend
        plt.tight_layout()

        plt.savefig("data/times.png", bbox_inches='tight')
            
def test_all_sizes(size_list: list) -> dict:
    results = dict()
    for size in tqdm(size_list, desc="Testing matrix sizes"):
        temp = get_test_results(size)
        for k in temp.keys():
            if not(k in results):
                results[k] = [temp[k]]
            else:
                results[k].append(temp[k])
    return results
    
def get_test_results(size : int):
    matrix = Matrix.randomMatrix(size, size)
    
    execution_times = {}

    execution_times["getDeterminant"] = get_time(matrix.getDeterminant)
    execution_times["getMinor"] = get_time(matrix.getMinor, 0, 0)
    execution_times["transpose"] = get_time(matrix.transpose)
    execution_times["inverse"] = get_time(matrix.inverse)
    execution_times["threshold_matrix"] = get_time(matrix.threshold_matrix, 0.5)
    execution_times["toInt"] = get_time(matrix.toInt)
    execution_times["toArray"] = get_time(matrix.toArray)
    execution_times["scipyEigenvalues"] = get_time(matrix.scipyEigenvalues)
    execution_times["numpyEigenvalues"] = get_time(matrix.numpyEigenvalues)
    execution_times["randomMatrix"] = get_time(Matrix.randomMatrix, size, size)
    execution_times["randintMatrix"] = get_time(Matrix.randintMatrix, size, size, 0, 10)
    execution_times["randomFloatMatrix"] = get_time(Matrix.randomFloatMatrix, size, size, 0.0, 1.0)
    execution_times["copy"] = get_time(matrix.copy)
    execution_times["__add__"] = get_time(lambda: matrix + matrix)
    execution_times["__sub__"] = get_time(lambda: matrix - matrix)
    execution_times["__mul__"] = get_time(lambda: matrix * matrix)
    execution_times["__eq__"] = get_time(lambda: matrix == matrix)
    execution_times["get_column"] = get_time(matrix.get_column, 0)
    execution_times["get_row"] = get_time(matrix.get_row, 0)
    execution_times["matrix_norm_L1"] = get_time(matrix.matrix_norm, 'L1')
    execution_times["matrix_norm_L2"] = get_time(matrix.matrix_norm, 'L2')
    execution_times["matrix_norm_Linf"] = get_time(matrix.matrix_norm, 'Linf')
    
    return execution_times

if __name__ == "__main__":
    main()
