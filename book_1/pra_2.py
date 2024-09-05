#クラスの復習

class Dog:
    def bark(self):
        print('woof')
        pass
    pass

# Dog().bark()
sizzles = Dog()
# sizzles.bark()
mutley = Dog()
# mutley.bark()

class Dog:
    def __init__(self, petname, temp):
        self.name = petname
        self.temperature = temp
        pass

    def status(self):
        print('dog name is', self.name)
        print('dog templature is', self.temperature)
        pass

    def setTemperature(self, temp):
        self.temperature = temp
        pass

    def bark(self):
        print('woof')
        pass
    pass

Iassie = Dog('Iassie', 37)
Iassie.status()
Iassie.setTemperature(40)
Iassie.status()