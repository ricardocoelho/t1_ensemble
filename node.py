class Node:
    def __init__(self, attribute=None, is_leaf=False, category=None):
        self.attribute = attribute
        self.is_leaf = is_leaf
        self.category = category
        self.gain = None
        self.child={}

    def set_category(self, category):
        self.category = category

    def set_leaf(self):
        self.is_leaf = True
    
    def set_attribute(self, attribute):
        self.attribute =  attribute

    def set_gain(self, gain):
        self.gain = gain

    def predict(self, inst):
        if self.is_leaf:
            return self.category

        if inst[self.attribute].item() in self.child:
            return self.child[inst[self.attribute].item()].predict(inst)
        else:
            return self.category

