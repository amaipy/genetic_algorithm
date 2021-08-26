import numpy as np
from random import random, randint

TERMINAL_RATE = 0.5

LOW_CONSTANT = -1
HIGH_CONSTANT = 1

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mult(x, y):
    return x * y

def div(x, y):
    if abs(y) <= 0.001:
        return x
    else:
        return x / y


FUNCTIONS = [add, sub, mult, div]

class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
    
    def node_data(self):
        if self.data in FUNCTIONS:
            return self.data.__name__
        else:
            return str(self.data)

    def execute_tree(self, variables):
        if (self.data in FUNCTIONS):
            return self.data(self.left.execute_tree(variables), self.right.execute_tree(variables))
        elif isinstance(self.data, str):
            for var, value in variables.items(): 
                if self.data == var:
                    return value
        else:
            return self.data
    
    def random_tree(self, variables, min_depth, grow, max_depth, depth=0):
        if depth < min_depth or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:
            if random() > TERMINAL_RATE:
                self.data = np.random.uniform(low=LOW_CONSTANT, high=HIGH_CONSTANT, size=1)[0]
            else:
                self.data = variables[randint(0, len(variables)-1)]
        else:
            if random() > 0.5:
                if random() > TERMINAL_RATE:
                    self.data = np.random.uniform(low=LOW_CONSTANT, high=HIGH_CONSTANT, size=1)[0]
                else:
                    self.data = variables[randint(0, len(variables)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]

        if self.data in FUNCTIONS:
            self.left = GPTree()
            self.left.random_tree(variables, min_depth, grow, max_depth, depth = depth + 1) 
            self.right = GPTree()
            self.right.random_tree(variables, min_depth, grow, max_depth, depth= depth + 1) 
    
    def mutation(self, rate, variables, min_depth):
        if random() < rate:
            self.random_tree(variables, min_depth, grow=True, max_depth=2)
        elif self.left: 
            self.left.mutation(rate, variables, min_depth)
        elif self.right: 
            self.right.mutation(rate, variables, min_depth)
    
    def size(self): 
        if isinstance(self.data, float): 
            return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def search_node(self, index):
        if index <= 1:
            return self.node_copy()
        l = self.left.size() if self.left else 0
        if (index < l):
            return self.left.search_node(index)
        if index == l:
            return self.left.node_copy()
        return self.right.search_node(index - l - 1)

    def swap_node(self, index, sub_tree):
        if index <= 1:
            self.left = sub_tree.left.node_copy() if sub_tree.left else None
            self.right = sub_tree.right.node_copy() if sub_tree.right else None
            self.data = sub_tree.data
            return 0
        l = self.left.size() if self.left else 0
        if index < l:
            return self.left.swap_node(index, sub_tree)
        if index == l:
            self.left.left = sub_tree.left.node_copy() if sub_tree.left else None
            self.left.right = sub_tree.right.node_copy() if sub_tree.right else None
            self.left.data = sub_tree.data
            return 0
        return self.right.swap_node(index - l - 1, sub_tree)

    def node_copy(self):
        t = GPTree()
        t.data = self.data
        if self.left:
            t.left = self.left.node_copy()
        if self.right:
            t.right = self.right.node_copy()
        return t
    
    def crossover(self, rate, other):
        if random() < rate:
            index_f = randint(1, self.size())
            index_s = randint(1, other.size())
            sub_tree = other.search_node(index_s)
            self.swap_node(index_f, sub_tree) 