import numpy as np
import matplotlib.pyplot as plt

# data
x = np.array([0, 2.50, 5.00, 7.50, 10.00])  # sr added (ppb)
y = np.array([28.0, 34.3, 42.8, 51.5, 58.6])  # signal (a.u.)
n = len(x)

# least squares fit
xbar, ybar = x.mean(), y.mean()
sxx = np.sum((x - xbar) ** 2)
m = np.sum((x - xbar) * (y - ybar)) / sxx
b = ybar - m * xbar

# r squared
resid = y - (m * x + b)
r2 = 1 - np.sum(resid ** 2) / np.sum((y - ybar) ** 2)

# x-intercept = concentration of unknown
x_int = -b / m
conc = abs(x_int)

# std dev of residuals (n-2 dof)
sy = np.sqrt(np.sum(resid ** 2) / (n - 2))

# uncertainty of x-intercept
ux = (sy / abs(m)) * np.sqrt(1 / n + ybar ** 2 / (m ** 2 * sxx))

# part c: ppb in 10.0 ml -> ppm in enamel
v_ml, mass_mg = 10.0, 0.750
ppm = conc * v_ml / mass_mg  # ng/mg = ppm
ppm_u = ux * v_ml / mass_mg

# print values
print(f"n = {n}")
print(f"xbar = {xbar:.4f}, ybar = {ybar:.4f}")
print(f"sum (xi - xbar)^2 = {sxx:.4f}")
print(f"slope m = {m:.4f}")
print(f"intercept b = {b:.4f}")
print(f"r^2 = {r2:.4f}")
print(f"sy = {sy:.4f}")
print(f"x-intercept = {x_int:.4f} ppb")
print(f"[sr] in solution = {conc:.4f} +/- {ux:.4f} ppb")
print(f"[sr] in enamel = {ppm:.1f} +/- {ppm_u:.1f} ppm")

# plot
xx = np.linspace(x_int, 10, 100)
plt.scatter(x, y, label="data")
plt.plot(xx, m * xx + b, label=f"y = {m:.3f}x + {b:.2f}")
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(x_int, color="red", ls="--", lw=0.8, label=f"x-int = {x_int:.2f}")
plt.xlabel("sr added (ppb)")
plt.ylabel("signal (a.u.)")
plt.legend()
plt.show()