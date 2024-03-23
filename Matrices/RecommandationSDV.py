from .Matrix import Matrix
from .Vector import Vector
    
def recommendedList(liked_movie_index, VT : Matrix, selected_movies_num) -> tuple:
    recommended = []
    for i in range(len(VT.rows)):
        if i != liked_movie_index:
            recommended.append([i, Vector(VT[i]).dotProduct(Vector(VT[liked_movie_index]))])
    
    recommended = tuple(sorted(recommended))
    return recommended[:selected_movies_num]