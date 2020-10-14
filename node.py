class Node:
    def __init__(self, attribute=None, is_leaf=False, target_value=None):
        self.attribute = attribute
        self.is_leaf = is_leaf
        self.target_value = target_value
        self.gain = None
        self.child={}

        self.numeric_condition= None

    def set_target_value(self, target_value):
        self.target_value = target_value

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
            return self.target_value

        # print("***atributo: ", self.attribute)
        # print("***instancia: ***\n", inst)
        # print("***inst[attrib] = ", inst[self.attribute])

        if self.numeric_condition:
            key = inst[self.attribute] <= self.numeric_condition
        else:
            key = inst[self.attribute]

        if key in self.child:
            return self.child[key].predict(inst)
        else:
            return self.target_value

