class Node:
    def __init__(self, attribute=None, is_leaf=False, category=None):
        self.attribute = attribute
        self.is_leaf = is_leaf
        self.category = category
        self.gain = None
        self.child={}

        self.numeric_condition= None

    def set_category(self, category):
        self.category = category

    def set_leaf(self):
        self.is_leaf = True
    
    def set_attribute(self, attribute):
        self.attribute =  attribute

    def set_gain(self, gain):
        self.gain = gain
    
    def set_numeric_condition(self, num):
        self.numeric_condition = num

    def predict(self, inst):
        if self.is_leaf:
            return self.category

        key = inst[self.attribute].item()
        if self.numeric_condition:
            key = inst[self.attribute].item() <= self.numeric_condition

        if key in self.child:
            return self.child[key].predict(inst)
        else:
            return self.category

