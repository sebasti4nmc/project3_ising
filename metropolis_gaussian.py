import numpy as np
from scipy.stats import norm
import random 
import matplotlib.pyplot as plt

mean, std_dev = 0, 1

def pdf_ratio(a, b, mean=mean, std_dev=std_dev):
    return norm.pdf(b, mean, std_dev) / norm.pdf(a, mean, std_dev)

def metropolis(x):
    dx = (random.random() * 2) - 1
    x_prop = x + dx
    ratio = pdf_ratio(x, x_prop)

    if random.random() < ratio:     # ratio >= 1 always accepts
        return x_prop
    return x

def main():
    x = (random.random() * 2) - 1
    print(x)
    xs = [x]

    for i in range(100000):
        x = metropolis(x)
        xs.append(x)

    plt.hist(xs, bins='auto', density=True, alpha=0.6, label="MCMC samples")

    # Overlay the true Gaussian
    grid = np.linspace(-4, 4, 400)
    plt.plot(grid, norm.pdf(grid, mean, std_dev), "r-", lw=2, label="true N(0,1) pdf")

    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show()

main()