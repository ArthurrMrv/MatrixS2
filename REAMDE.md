# Matrices Module

The Matrices module provides classes and functions for working with matrices and vectors in Python. It offers functionalities such as matrix operations, linear algebra operations, and utilities for working with matrices and vectors.


## Usage
Here's a brief overview of the main classes and functions available in the Matrices module:

### Matrix Class

The `Matrix` class represents a mathematical matrix. It provides functionalities for matrix operations, such as addition, subtraction, multiplication, transpose, inverse, and more.

#### Example Usage:

```python
from Matrices import Matrix

# Create a matrix
A = Matrix([[1, 2], [3, 4]])

# Get matrix shape
print("Matrix Shape:", A.get_shape())

# Compute matrix determinant
print("Determinant:", A.getDeterminant())
```

### Vector Class

The `Vector` class represents a mathematical vector, which is a specialized form of a matrix. It provides functionalities specific to vectors, such as normalization, dot product calculation, and element-wise division.

#### Example Usage:

```python
from Matrices import Vector

# Create a vector
v = Vector([[1], [2], [3]])

# Normalize the vector
v.normalize()
print("Normalized Vector:", v)

# Compute dot product between vectors
v1 = Vector([[1], [2], [3]])
v2 = Vector([[4], [5], [6]])
dot_product = Vector.dotProduct(v1, v2)
print("Dot Product:", dot_product)
```

### Functions

The Matrices module also includes various functions for matrix operations, linear algebra, and matrix generation.

#### Example Usage:

```python
from Matrices import qr_algorithm, qr_decomposition

# Perform QR decomposition
Q, R = qr_decomposition(A)

# Find eigenvalues and eigenvectors using QR algorithm
eigenvalues, eigenvectors = qr_algorithm(A)
```

## Contributing

Contributions to the Matrices module are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ArthurrMrv/MatrixS2.git).