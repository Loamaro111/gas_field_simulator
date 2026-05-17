from dataclasses import dataclass

@dataclass
class NodeState:
    name: str              # идентификатор элемента ("well_1", "shlyf", "dcs")
    P_in: float            # давление на входе [атм]
    P_out: float           # давление на выходе [атм]
    dP: float              # перепад давления [атм]
    q_std: float           # коммерческий расход [ст.м³/сут]
    q_res: float | None    # объёмный расход при местных условиях [м³/сут]
    v: float | None        # скорость потока [м/с]
    rho: float | None      # плотность газа [кг/м³]