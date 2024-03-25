import matplotlib.pyplot as plt
from Matrices import Matrix, Vector, Functions as f
import numpy as np
import time

def main ():
    # Create a matrix
    matrix = Matrix.randomMatrix(100, 100)
    # Plot the singular values of the SVD
    plot_singular_values(matrix)
    
    # plot_eigenvalues_comparison(matrix)
    # plot_linear_system_solver_comparison(100)
    
# Define a function to generate and plot the singular values Sigma of the SVD
def get_time(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

def plot_singular_values(matrix):
    _, Sigma, _ = f.getSVD_qr(matrix)
    plt.plot(Sigma, marker='o')
    plt.title('Singular Values of the SVD')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig('data/singular_values.png')


def plot_eigenvalues_comparison(A):

    your_eigenvalues = f.qr_algorithm(A)[0]
    numpy_eigenvalues = np.linalg.eigvals(A.toArray())

    plt.figure(figsize=(8, 6))
    plt.plot(your_eigenvalues, label='Your QR Algorithm')
    plt.plot(numpy_eigenvalues, label='NumPy Eig')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Comparison of Eigenvalues from QR Algorithm')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/eigenvalues_comparison.png')
    plt.close()

def plot_linear_system_solver_comparison(matrix_size):
    A = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size)

    your_solution = f.solve_linear_system(Matrix(A), Vector(b))
    numpy_solution = np.linalg.solve(A, b)

    differences = np.abs(your_solution - numpy_solution)

    plt.figure(figsize=(8, 6))
    plt.plot(differences)
    plt.xlabel('Index')
    plt.ylabel('Difference in Solutions')
    plt.title('Comparison of Solutions from Linear System Solver')
    plt.grid(True)
    plt.savefig('data/linear_system_solver_comparison.png')
    plt.close()



# Example usage
if __name__ == "__main__":
    main()
