import time

class timer():
    def __init__(self):
        self.acc = 0
        self.time_current()

    def time_current(self):
        self.t0 = time.time()

    def time_pass(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.time_pass()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0
