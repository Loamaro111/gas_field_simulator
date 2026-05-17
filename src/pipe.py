import math
from src.state import NodeState
from src.fluid import Fluid


class Pipe:
    def __init__(self, L: float, D: float, roughness: float, fluid: Fluid, vertical_depth: float = 0.0, name: str = "pipe"):
        self.L = L
        self.D = D
        self.roughness = roughness
        self.fluid = fluid
        self.vertical_depth = vertical_depth
        self.name = name

    def dp(self, P: float, q_std: float) -> NodeState:
        if q_std <= 0:
            rho = self.fluid.ro(P)
            dP_stat = (rho * 9.81 * self.vertical_depth) / 101325.0
            return NodeState(
                name=self.name,
                P_in=P,
                P_out=P - dP_stat,
                dP=dP_stat,
                q_std=0.0,
                q_res=0.0,
                v=0.0,
                rho=rho
            )

        rho = self.fluid.ro(P)
        bg = self.fluid.bg(P)
        mu = self.fluid.mu(P) / 1000.0

        q_res = q_std * bg
        v = (4.0 * q_res) / (math.pi * self.D ** 2 * 86400.0)

        Re = (rho * v * self.D) / mu

        if Re < 2300:
            lmd = 64.0 / Re
        else:
            lmd = 0.02
            for _ in range(50):
                term = (self.roughness / (3.7 * self.D)) + (2.51 / (Re * math.sqrt(lmd)))
                lmd_new = (-2.0 * math.log10(term)) ** -2
                if abs(lmd_new - lmd) < 1e-6:
                    lmd = lmd_new
                    break
                lmd = lmd_new

        dP_fric = lmd * (self.L / self.D) * (rho * v ** 2 / 2.0)
        dP_stat = rho * 9.81 * self.vertical_depth

        dP_atm = (dP_fric + dP_stat) / 101325.0

        return NodeState(
            name=self.name,
            P_in=P,
            P_out=P - dP_atm,
            dP=dP_atm,
            q_std=q_std,
            q_res=q_res,
            v=v,
            rho=rho
        )