import numpy as np
import pandas as pd
from src.interpolator import LinearInterpolator


class Fluid:
    def __init__(self, M: float, rho_c: float, xa: float, xy: float, T: float):
        self.M = M
        self.rho_c = rho_c
        self.xa = xa
        self.xy = xy
        self.T = T

        self.x_eq = 1.0 - xa - xy
        self.z_c = 1.0 - (0.0741 * rho_c - 0.006 - 0.063 * xa - 0.0575 * xy) ** 2
        self.M3 = (24.05525 * self.z_c * rho_c - 28.0135 * xa - 44.01 * xy) / self.x_eq
        self.H = 128.64 + 47.479 * self.M3

        try:
            df = pd.read_csv('interp_data.csv', sep=',')
            df.columns = df.columns.str.strip()
            _ = df['pressure, atm']
        except KeyError:
            df = pd.read_csv('interp_data.csv', sep=';')
            df.columns = df.columns.str.strip()

        self.visc_interp = LinearInterpolator(
            df['pressure, atm'].tolist(),
            df['viscosity, cP'].tolist()
        )

    def _B1(self) -> float:
        T = self.T
        return (-0.425468 + 2.865e-3 * T - 4.62073e-6 * T ** 2 +
                (8.77118e-4 - 5.56281e-6 * T + 8.81514e-9 * T ** 2) * self.H +
                (-8.24747e-7 + 4.31436e-9 * T - 6.08319e-12 * T ** 2) * self.H ** 2)

    def _B2(self) -> float:
        T = self.T
        return -0.1446 + 0.74091e-3 * T - 0.91195e-6 * T ** 2

    def _B23(self) -> float:
        T = self.T
        return -0.339693 + 0.161176e-2 * T - 0.204429e-5 * T ** 2

    def _B3(self) -> float:
        T = self.T
        return -0.86834 + 0.40376e-2 * T - 0.51657e-5 * T ** 2

    def _C1(self) -> float:
        T = self.T
        return (-0.302488 + 1.95861e-3 * T - 3.16302e-6 * T ** 2 +
                (6.46422e-4 - 4.22876e-6 * T + 6.88157e-9 * T ** 2) * self.H +
                (-3.32805e-7 + 2.2316e-9 * T - 3.67713e-12 * T ** 2) * self.H ** 2)

    def _C2(self) -> float:
        T = self.T
        return 7.8498e-3 - 3.9895e-5 * T + 6.1187e-8 * T ** 2

    def _C3(self) -> float:
        T = self.T
        return 2.0513e-3 + 3.4888e-5 * T - 8.3703e-8 * T ** 2

    def _C223(self) -> float:
        T = self.T
        return 5.52066e-3 - 1.68609e-5 * T + 1.57169e-8 * T ** 2

    def _C233(self) -> float:
        T = self.T
        return 3.58783e-3 + 8.06674e-6 * T - 3.25798e-8 * T ** 2

    def _B_star(self) -> float:
        T = self.T
        return 0.72 + 1.875e-5 * (320.0 - T) ** 2

    def _C_star(self) -> float:
        T = self.T
        return 0.92 + 0.0013 * (T - 270.0)

    def z(self, P: float) -> float:
        P_MPa = P * 0.101325

        B1 = self._B1()
        B2 = self._B2()
        B23 = self._B23()
        B3 = self._B3()
        C1 = self._C1()
        C2 = self._C2()
        C223 = self._C223()
        C233 = self._C233()
        C3 = self._C3()
        B_star = self._B_star()
        C_star = self._C_star()

        Z13 = -0.865
        Y12 = 0.92
        Y13 = 0.92
        Y123 = 1.1

        term1 = self.x_eq ** 2 * B1
        term2 = self.x_eq * self.xa * B_star * (B1 + B2)
        term3 = -1.73 * self.x_eq * self.xy * np.sqrt(B1 * B3)
        term4 = self.xa ** 2 * B2
        term5 = 2.0 * self.x_eq * self.xy * B23
        term6 = self.xy ** 2 * B3
        Bm = term1 + term2 + term3 + term4 + term5 + term6

        C1_2 = C1 ** 2
        C2_2 = C2 ** 2
        C3_2 = C3 ** 2
        CA112 = C1_2 * C2
        CA113 = C1_2 * C3
        CA122 = C1 * C2_2
        CA123 = C1 * C2 * C3
        CA133 = C1 * C3_2

        cbrt112 = np.cbrt(CA112)
        cbrt113 = np.cbrt(CA113)
        cbrt122 = np.cbrt(CA122)
        cbrt123 = np.cbrt(CA123)
        cbrt133 = np.cbrt(CA133)

        Cm = (self.x_eq ** 3 * C1 +
              3.0 * self.x_eq ** 2 * self.xa * cbrt112 * C_star +
              3.0 * self.x_eq ** 2 * self.xy * cbrt113 * Y13 +
              3.0 * self.x_eq * self.xa ** 2 * cbrt122 * C_star +
              6.0 * self.x_eq * self.xa * self.xy * cbrt123 * Y123 +
              3.0 * self.x_eq * self.xy ** 2 * cbrt133 * Y13 +
              self.xa ** 3 * C2 +
              3.0 * self.xa ** 2 * self.xy * C223 +
              3.0 * self.xa * self.xy ** 2 * C233 +
              self.xy ** 3 * C3)

        b = 1000.0 * P_MPa / (2.7715 * self.T)

        B0 = b * Bm
        C0 = b ** 2 * Cm

        A1 = 1.0 + B0
        A0 = 1.0 + 1.5 * (B0 + C0)

        D = A0 ** 2 - A1 ** 3
        if D < 0:
            return np.nan

        sqrtD = np.sqrt(D)
        A2 = np.cbrt(A0 - sqrtD)
        z_factor = (1.0 + A2 + A1 / A2) / 3.0
        return z_factor

    def ro(self, P: float) -> float:
        P_pa = P * 101325
        R = 8.314
        return (P_pa * self.M) / (self.z(P) * R * self.T)

    def bg(self, P: float) -> float:
        return (101325 / (P * 101325)) * (self.T / 293.15) * self.z(P)

    def mu(self, P: float) -> float:
        return self.visc_interp.predict(P)
