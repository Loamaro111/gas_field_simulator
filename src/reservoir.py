from dataclasses import dataclass
from src.fluid import Fluid


@dataclass
class ResProps:
    P: float
    V: float
    T: float


class Reservoir:
    def __init__(self, resprops: ResProps, fluid: Fluid):
        self.resprops = resprops
        self.fluid = fluid

    def p2(self, q_total: float, dt: float = 1.0) -> float:
        """Расчет нового пластового давления через шаг dt"""
        P_curr = self.resprops.P
        V_res = self.resprops.V

        Z = self.fluid.z(P_curr)
        rho_res = self.fluid.ro(P_curr)

        # Плотность при стандартных условиях (P = 1 атм, T = 293.15 K, Z ≈ 1)
        rho_std = (101325.0 * self.fluid.M) / (1.0 * 8.314 * 293.15)

        # Уравнение материального баланса
        dP = (Z * rho_std / rho_res) * (q_total / V_res) * dt

        return P_curr - dP
