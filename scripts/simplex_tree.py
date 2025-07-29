from typing import Dict
#from dataclasses import dataclass
from itertools import combinations
from copy import copy

#@dataclass
#class Simplex:
#    def __init__(self, k=-1, data=[]):
#        assert len(data) == k + 1 
#        self.k : int = k
#        self.data = data
#        #self.label = label
#

# Topological construction with strict and simple rules
# i.) Any sub-simplex of a simplex in the complex is also an element of the complex
# ii.) Non-empty intersection (meet) of any 2 simplices is sub-simplex of both simplices
class SimplicialComplex:
    def __init__(self, max_grade:int=-1, simplex_store=None):
        self.max_grade = max_grade
        self.k_simplex_store = simplex_store if simplex_store is not None else {}

    # Any time simplex is added, recursively add all sub-simplex until 0-simplex
    # Currently stores multiple copies of data; should store one copy and make all
    #   higher order simplices out of pointers to data
    # Currently assumes all given data is a full simplex, which is absurd
    def add_simplex(self, data) -> None:
        grade = len(data) - 1

        if grade > self.max_grade:
            self.max_grade = grade

        while grade > -1:
            k_simplex_list = self.k_simplex_store.get(grade, [])
            for d_c in combinations(data, grade+1):
                if d_c not in k_simplex_list:
                    k_simplex_list.append(d_c)
            self.k_simplex_store[grade] = k_simplex_list 
            grade -= 1

        return
         
    # Find intersection of self complex and other complex
    # Could return empty simplex: (max_grade=-1, simplex_store={})
    # Inductive approach to meet:
    #   Starts from 0-simplex; First empty level implies no further intersection
    #   Follows from downward closure of sub-simplices
    def meet(self, other_complex):
        possible_max_grade = min(self.max_grade, other_complex.max_grade)

        grade = -1
        simplex_store = {}
        while grade < possible_max_grade:
            #temp_list = []
            #temp_list.extend(self.k_simplex_store[grade+1])
            #temp_list.extend(other_complex.k_simplex_store[grade+1])
            #temp_list  = list(set.intersection(*map(set, temp_list))) 
            common = list(set(self.k_simplex_store[grade+1]) & set(other_complex.k_simplex_store[grade+1])) 
            if len(common) == 0:
                break

            grade += 1
            simplex_store[grade] = common

        return SimplicialComplex(grade, simplex_store)

    # Finds the join of 2 simplicial complexes
    # Returns another simplicial complex
    # Relies first on the disjoint union
    # adds additional simplices to (complete a topology)(??)
    def join(self, other_complex):
        print("IMPLEMENT")
        return -1
        #max_grade = max(self.max_grade, other_complex.max_grade)
        #joint = SimplicialComplex(grade=max_grade)

        #for self_simplex in self.k_simplex_store.values():
        #    for other_simplex in other.k_simplex_store.values():
        #        for s1 in self_simplex:
        #            for s2 in other_simplex:
        #                if set(s1).is_disjoint(s2):

if __name__ == "__main__":
    sc = SimplicialComplex()

    sc.add_simplex([3, 5, 9 ,7])
    sc.add_simplex([3, 8, 10])
    sc.add_simplex([5, 8])
    print("SC")
    print("k store:", sc.k_simplex_store)
    print("sc max grade", sc.max_grade)

    other = SimplicialComplex()
    other.add_simplex([1, 2, 3, 4, 5])

    print("other max grade: ", other.max_grade)
    print("other simplex store: ", other.k_simplex_store)

    intersection = sc.meet(other)

    print("int max grade: ", intersection.max_grade)
    print("int simplex store: ", intersection.k_simplex_store)
