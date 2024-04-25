import time

class PI:
    def __init__(self, p, i) -> None:
        self.p = p
        self.i = i
        self.errors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.previous_time = 0

    def calculate_error(self, error):
        self.errors.pop(0)
        self.errors.append(error)

        proportional = error*self.p

        integral = 0
        for e in self.errors:
            integral += e*self.i

        return proportional + integral