class PI:
    def __init__(self, p_gain, i_gain) -> None:
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.errors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def compute_control(self, error):
        self.errors.pop(0)
        self.errors.append(error)

        error_integral = 0.0
        for e in self.errors:
            error_integral += e

        control = error*self.p_gain + error_integral*self.i_gain  # PI controller
        return control
    