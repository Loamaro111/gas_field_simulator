class LinearInterpolator:
    def __init__(self, xs: list, ys: list):
        if len(xs) != len(ys):
            raise ValueError("Списки xs и ys должны быть одинаковой длины")
        self.xs = xs
        self.ys = ys

    def predict(self, xp: float) -> float:
        if xp < self.xs[0] or xp > self.xs[-1]:
            raise ValueError(f"Точка {xp} вне диапазона интерполяции [{self.xs[0]}, {self.xs[-1]}]")

        # Поиск интервала
        for i in range(len(self.xs) - 1):
            if self.xs[i] <= xp <= self.xs[i + 1]:
                # Формула линейной интерполяции
                dy = self.ys[i + 1] - self.ys[i]
                dx = self.xs[i + 1] - self.xs[i]
                if dx == 0:
                    return self.ys[i]
                return self.ys[i] + dy * (xp - self.xs[i]) / dx

        return self.ys[-1]