import sys
from flask import Flask, render_template
from flask import Flask, request, jsonify
import sys
import os
import numpy as np


# Add the directory containing the Matrix module to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import Matrices as mat

# Create a Flask application instance
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/call_function', methods=['POST'])
def receive_data():
    data = request.json  # Get JSON data from the request
    # Process the data here
    # For example, you can access data["matrices"] and data["method"]
    ans = dict()
    
    ans["type"], ans["content"] = analyse_function_call(data)
    if type(ans["content"]) == np.ndarray:
        ans["content"] = ans["content"].tolist()
    response_data = ans
    return jsonify(response_data)

def analyse_function_call(data):
    matrices = matrices_to_np(data)
    match data["method"]:
        case "add":
            if len(matrices) < 2:
                return  ('text', "Error: At least two matrices are required for addition")
            return ('matrix', np.add(matrices[0], matrices[1]))
        case "subtract":
            if len(matrices) < 2:
                return ('text', "Error: At least two matrices are required for substraction")
            return ('matrix',np.subtract(matrices[0], matrices[1]))
        case "multiply":
            if len(matrices) < 2:
                return ('text', "Error: At least two matrices are required for multiplication")
            return ('matrix', np.matmul(matrices[0], matrices[1]))
        case "transpose":
            if len(matrices) > 1:
                return ('text', "Error: Only one matrix is required for transpose")
            return ('matrix', np.transpose(matrices[0]))
        case "determinant":
            if len(matrices) > 1:
                return ('text', "Error: Only one matrix is required for determinant")
            return ('number', np.linalg.det(matrices[0]))
        case "inverse":
            if len(matrices) > 1:
                return ('text', "Error: Only one matrix is required for inverse")
            return ('matrix', np.linalg.inv(matrices[0]))
        case "eigenvalues":
            if len(matrices) > 1:
                return ('text', "Error: Only one matrix is required for eigenvalues")
            return ('text', np.linalg.eig(matrices[0])[0])
        case "eigenvectors":
            if len(matrices) > 1:
                return ('text', "Error: Only one matrix is required for eigenvectors")
            return ('list[matrix]', [[[e] for e in m ] for m in np.linalg.eig(matrices[0])[1]])
        case _:
            return ('text', "Error: Invalid method")
        
def matrices_to_np(data):
    matrices = []
    for matrix in data["matrices"]:
        mat_data = [[float(e) for e in row] for row in data["matrices"][matrix]]
        matrices.append(np.array(mat_data))
    return matrices

# Run the Flask application
if __name__ == '__main__':
    app.run(host=sys.argv[1], port=int(sys.argv[2]), debug=True)
    
