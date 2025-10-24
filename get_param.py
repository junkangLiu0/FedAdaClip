import math

if __name__ == "__main__":
    q = 0.1
    delta = 1e-5
    sq5 = math.sqrt(5)

    epsilon = 1
    max_grad_norm = 1.0

    noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    sigma_gaussian = noise_multiplier * max_grad_norm

    alpha = sigma_gaussian**2 * math.log(1/q) / 2
    epsilon_low = q**2 * 6 * alpha / sigma_gaussian**2

    print(f"\nnoise multiplier ≥ {noise_multiplier}")
    print(f"\n1 ≤ alpha ≤ {alpha}")
    print(f"\nsigma_gaussian ≥ {sigma_gaussian} and ≥ {sq5}")
    print(f"\nepsilon ≥ {epsilon_low}")
