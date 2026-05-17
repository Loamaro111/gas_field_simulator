class DCS:
    def __init__(self, CR: float, P_line: float, q_ext: float = 0.0):
        self.CR = CR
        self.P_line = P_line
        self.q_ext = q_ext

    def P_in(self) -> float:
        """Давление на входе в ДКС [атм]"""
        if self.CR <= 1.0:
            return self.P_line
        return self.P_line / self.CR