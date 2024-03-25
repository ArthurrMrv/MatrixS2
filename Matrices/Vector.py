from .Matrix import Matrix
from typing import Iterable

class Vector(Matrix):
    def __init__(self, values: list[list], *, mutable=False) -> None:
        
        if isinstance(values, Iterable) and not isinstance(values[0], Iterable):
            values = tuple(map(lambda x : [x], values))
    
        Matrix.__init__(self, values, mutable=mutable)
    
        if 1 not in Vector.get_shape(self):
            raise ValueError("The vector must be a row or column vector (got a a shape of {})".format(Vector.get_shape(self)))
    
        self.ColVector : bool = self.get_shape()[1] == 1
    
    def isColVector(self) -> bool:
        return self.ColVector
    
    def normalize(self):
        x_norm = max([abs(v) for v in self.values])
        self = Vector([[elem / x_norm for elem in l]  for l in self.get_values()])
        
    def copy(self) -> 'Vector':
        return Vector(self.get_values())
    
    def norm(self) -> float:
        return sum(self.values[i*self.getNbCol() + 0]**2 for i in range(self.getNbRows()))**0.5
    
    def transpose(self) -> 'Vector':
        return Vector(super().transpose())
    
    @classmethod
    def dotProduct(cls, *vectors : list) -> float:
        zippedShapes = tuple(zip(*map(Vector.get_shape, vectors)))
        if len(set(zippedShapes[0])) != 1 or len(set(zippedShapes[1])) != 1:
            raise ValueError("Vectors must have the same shape")
        del zippedShapes
        
        if not(vectors[0].isColVector()):
            vectors = tuple(map(Vector.transpose, vectors))
        
        ans = 0
        for i in range(1, len(vectors)):
            mult = 1
            for v in vectors:
                mult *= v[i, 0]
            ans += mult
        return ans
    
    def __mul__(self, other) -> 'Matrix':
        
        return Vector(Matrix.__mul__(self, other).get_values())
        
    def __sub__(self, other) -> Matrix:
        return Vector(Matrix.__sub__(self, other).get_values())
    
    def __getitem__(self, index):
        if type(index) == tuple:
            return self.get_values()[index[0]][index[1]]
        elif self.isColVector():
            return self.get_values()[index][0]
        else:
            return self.get_values()[0][index]
        
    def __iter__(self):
        for i in range(max(self.get_shape())):
            yield self[i]
            
    def div(self, other : int) -> 'Vector':
        return Vector([[elem / other for elem in l]  for l in self.get_values()])
