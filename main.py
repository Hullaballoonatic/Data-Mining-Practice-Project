import string

def isAttribute(line):
  return line[0] in string.ascii_letters

def isContinuous(line):
  return ',' in line

def isCategoricalAttribute(line):
  return isAttribute(line) and !isContinuous(line)

def __main__():
  nameFile = open('adult.names', "r")
  attributeLines = filter(isContinuousAttribute, nameFile.readLines())
  namesToCategories = [ line.split(': ') for line in attributeLines ]
  attributes = [ Attribute(name = line.split(': ')[0], categories = line.split(': ')[1].split(', ')) for line in filter(!isContinuous, nameLines) ]
  
  dataFile = open('adult.data', "r")
  data = [ line.split(", ") for line in dataFile.readlines() ]
  dataFile.close()

  testFile = open('adult.test', "r")
  test = [ line.split(", ") for line in testFile.readLines() ]
  testFile.close()

class Attribute: 
  def __init__(self, name, categories = None):
    self.categories = categories
    self.name = name

  def __isContinuous__(self):
    return self.categories == None

  def __isCategorical(self):
    return self.categories != None