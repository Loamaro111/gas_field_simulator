import pandas as pd
from scipy.optimize import root
from src.state import NodeState
from src.reservoir import Reservoir
from src.compressor import DCS


class FieldSimulator:
    def __init__(self, reservoir: Reservoir, wells: list, shlyf, dcs: DCS):
        self.reservoir = reservoir
        self.wells = wells
        self.shlyf = shlyf
        self.dcs = dcs

    def solve(self, P_res: float) -> dict[str, NodeState]:
        """Расчет рабочей точки системы (дебиты и давления)"""

        def residual(x):
            q1, q2, q3, P_man = x

            # Искусственное ограничение для стабильности решателя
            q1_c = max(0.1, q1)
            q2_c = max(0.1, q2)
            q3_c = max(0.1, q3)
            P_man_c = max(1.0, P_man)

            # Забойные давления через VLP
            P_bhp1 = P_man_c + self.wells[0].pipe.dp(P_man_c, q1_c).dP
            P_bhp2 = P_man_c + self.wells[1].pipe.dp(P_man_c, q2_c).dP
            P_bhp3 = P_man_c + self.wells[2].pipe.dp(P_man_c, q3_c).dP

            # Уравнения притока IPR
            eq1 = q1 - self.wells[0].q(P_res, P_bhp1)
            eq2 = q2 - self.wells[1].q(P_res, P_bhp2)
            eq3 = q3 - self.wells[2].q(P_res, P_bhp3)

            # Баланс давлений на манифолде (Шлейф + ДКС)
            q_total = q1_c + q2_c + q3_c + self.dcs.q_ext
            P_in_dcs = self.dcs.P_in()
            eq4 = P_man - (P_in_dcs + self.shlyf.dp(P_man_c, q_total).dP)

            return [eq1, eq2, eq3, eq4]

        # Начальное приближение
        x0 = [500.0, 500.0, 500.0, self.dcs.P_in() + 5.0]

        # Численное решение системы
        sol = root(residual, x0, method='hybr')
        q1_res, q2_res, q3_res, P_man_res = sol.x

        # Обработка остановки скважин
        q1_res = max(0.0, q1_res)
        q2_res = max(0.0, q2_res)
        q3_res = max(0.0, q3_res)
        P_man_res = max(1.0, P_man_res)

        # Формирование финальных состояний
        P_bhp1 = P_man_res + self.wells[0].pipe.dp(P_man_res, q1_res).dP
        P_bhp2 = P_man_res + self.wells[1].pipe.dp(P_man_res, q2_res).dP
        P_bhp3 = P_man_res + self.wells[2].pipe.dp(P_man_res, q3_res).dP

        q_total = q1_res + q2_res + q3_res + self.dcs.q_ext

        ns_w1 = NodeState("well_1", P_res, P_bhp1, P_res - P_bhp1, q1_res, None, None, None)
        ns_w2 = NodeState("well_2", P_res, P_bhp2, P_res - P_bhp2, q2_res, None, None, None)
        ns_w3 = NodeState("well_3", P_res, P_bhp3, P_res - P_bhp3, q3_res, None, None, None)

        ns_shlyf = self.shlyf.dp(P_man_res, q_total)
        ns_shlyf.P_in = P_man_res
        ns_shlyf.P_out = P_man_res - ns_shlyf.dP
        ns_shlyf.name = "shlyf"

        P_in_dcs = self.dcs.P_in()
        ns_dcs = NodeState("dcs", P_in_dcs, self.dcs.P_line, P_in_dcs - self.dcs.P_line, q_total, None, None, None)

        return {
            "well_1": ns_w1,
            "well_2": ns_w2,
            "well_3": ns_w3,
            "shlyf": ns_shlyf,
            "dcs": ns_dcs
        }

    def run(self, N_days: int, dt: float = 1.0) -> pd.DataFrame:
        """Динамическая симуляция во времени"""
        results = []
        Gp = 0.0

        for t in range(N_days):
            P_res_curr = self.reservoir.resprops.P
            states = self.solve(P_res_curr)

            q1 = states["well_1"].q_std
            q2 = states["well_2"].q_std
            q3 = states["well_3"].q_std
            P_man = states["shlyf"].P_in

            q_wells_total = q1 + q2 + q3

            # Шаг во времени: обновление давления
            P_new = self.reservoir.p2(q_total=q_wells_total, dt=dt)
            self.reservoir.resprops.P = P_new

            # Накопленная добыча в тыс. ст. м3
            Gp += (q_wells_total * dt) / 1000.0
            results.append({
                "t": t * dt,
                "P_res": P_res_curr,
                "P_man": P_man,
                "q1": q1,
                "q2": q2,
                "q3": q3,
                "q_total": q_wells_total,
                "Gp": Gp
            })
            if t % 30 == 0 or t == N_days - 1:
                print(f"День {t:3d}: P_res = {P_res_curr:.2f} атм, q_total = {q_wells_total:.1f} ст.м³/сут, Gp = {Gp:.1f} тыс.м³")
        return pd.DataFrame(results)

