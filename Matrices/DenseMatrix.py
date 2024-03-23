from .Matrix import Matrix

class DenseMatrix(Matrix):
    def __init__(self, values: list[list]) -> None:
        
        self._nbRows = len(values)
        self._nbColumns = len(values[0])
        self.values = {(r, c) : values[r][c] for c in range(self._nbColumns) for r in range(self._nbRows) if values != 0}

        Matrix.__init__(self, values, mutable=False)

    def toTuple(self):
        return (self.values[(r, c)] if (r, c) in self.values else 0 for c in range(self._nbColumns) for r in range(self._nbRows))