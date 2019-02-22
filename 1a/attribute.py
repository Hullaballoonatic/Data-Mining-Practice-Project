class CategoricalAttribute:
    def __init__(self, name, categories, values=[]):
        self.name = name
        self.strWidth = len(max([name, *categories], key=len))
        self.values = []
        self.categories = categories
        self.occurrencesOfCategory = zip(categories, [0] * len(categories))

    def addValue(self, v):
        if (v in self.categories):
            self.values.append(v)
            self.occurrencesOfCategory[v] += 1

    def __str__(self):
        return "{0}: {1}\n".format(self.name, ", ".join(self.categories))
