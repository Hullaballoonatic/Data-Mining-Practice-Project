class Entry:
    def __init__(self, values, conditionalValue: bool):
        self.values = values
        self.conditionalValue = conditionalValue

    @property
    def numValues(self):
        return len(self.values)
