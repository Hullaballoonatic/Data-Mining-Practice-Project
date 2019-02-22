from attribute import CategoricalAttribute
from table import Table
from entry import Entry


def addAllEntries(file, table):
    for line in file.readlines():
        tmp = [x.strip() for x in line.split(',')]
        values = tmp[:-1]
        conditionalValue = tmp[-1] == '>50K'
        table.entries.append(Entry(values, conditionalValue))


attributeInfo = []

with open('1a/adult.names') as file:
    for line in file.readlines():
        tmp = line.split(': ')
        name = tmp[0]
        categories = tmp[1].split(', ')
        attributeInfo.append(CategoricalAttribute(name, categories))

data = Table('data', attributeInfo)
test = Table('test', attributeInfo)

with open('1a/adult.data') as file:
    addAllEntries(file, data)

with open('1a/adult.test') as file:
    addAllEntries(file, test)

with open('1a/out.data', 'w') as file:
    file.write(str(data))

with open('1a/out.test', 'w') as file:
    file.write(str(test))
