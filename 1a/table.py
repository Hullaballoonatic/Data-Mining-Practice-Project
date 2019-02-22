from attribute import CategoricalAttribute
from entry import Entry


class Table:
    def __init__(self, name, attributes: [CategoricalAttribute]):
        self.name = name
        self.attributes = attributes
        self.colWidth = max([attribute.strWidth for attribute in attributes]) + 2

    entries = []

    def addEntry(self, v: Entry):
        self.entries.append(v)
        for i in range(v.numValues):
            self.attributes[i].addValue(v.values[i])

    @property
    def numEntries(self):
        return len(self.entries)

    def __str__(self):
        prependLen = max(len(self.name), len(str(self.numEntries))) + 5
        result = self.name.upper().center(prependLen, ' ') + '|' + '|'.join([attribute.name.center(self.colWidth, ' ') for attribute in self.attributes]) + '\n'
        result += '-' * prependLen + '+' + '+'.join(['-' * self.colWidth] * len(self.attributes)) + '\n'
        for i in range(self.numEntries):
            entry = self.entries[i]
            result += str(i + 1).rjust(prependLen - 3) + ' {0} |'.format('*' if entry.conditionalValue else ' ')
            result += '|'.join([value.center(self.colWidth, ' ') for value in entry.values]) + '\n'
        return result
