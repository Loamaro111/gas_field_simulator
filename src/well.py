import math
from src.fluid import Fluid
from src.pipe import Pipe


class Well:
    def __init__(self, fluid: Fluid, k: float, h: float, re: float, rw: float, pipe: Pipe = None):
        self.fluid = fluid
        self.k = k  # Проницаемость, мД
        self.h = h  # Мощность, м
        self.re = re  # Радиус контура, м
        self.rw = rw  # Радиус скважины, м
        self.pipe = pipe  # Объект НКТ
        self.beta = 0.00852702

    def q(self, P_res: float, P_bhp: float) -> float:
        """Расчет дебита скважины по закону Дарси [ст.м³/сут]"""
        if P_res <= P_bhp:
            return 0.0

        mu = self.fluid.mu(P_res)

        # Коэффициент продуктивности C
        C = (self.beta * self.k * self.h) / (mu * math.log(self.re / self.rw))

        return C * (P_res - P_bhp)