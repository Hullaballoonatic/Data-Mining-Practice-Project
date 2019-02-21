class CategoricalAttribute:
  def __init__(self, name, categories):
    self.name = name
    self.categories = categories

  def __str__(self):
    return "{0}: {1}\n".format(self.name, ", ".join(self.categories))

class Table:
  def __init__(self, attributeInfo):
    self.attributeInfo = attributeInfo
  
  entries = []

with open('adult.names') as file:
  attributeInfo = [ CategoricalAttribute(line.split(': ')[0], line.split(': ')[1].split(', ')) for line in file.readlines() ]

data = Table(attributeInfo)
test = Table(attributeInfo)

with open('adult.data') as file:
  for entry in [ line.split(", ")[:-1] for line in file.readlines() ]:
    data.entries.append(entry)

with open('adult.test') as file:
  for entry in [ line.split(", ")[:-1] for line in file.readlines() ]:
    test.entries.append(entry)