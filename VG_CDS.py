import numpy as np
from scipy.special import gamma
from scipy.stats import norm
from scipy.integrate import quad

################################
### This is implementation of Fiorani, Luciano, Semeraro (2007) model
### for company debt valuation and CDS pricing.
### Purely theoretical, but involves complicated maths.
################################

class VarianceGammaCDS:

    def __init__(self, V, F, r, q, sigma, alpha, theta, T, ytm, period):
        self.V = V              # Initial firm assets
        self.F = F              # Face value of firms debt in the form of zero-coupon bond
        self.r = r              # Risk-free interest rate
        self.q = q              # Dividend yield
        self.sigma = sigma      # Volatility
        self.alpha = alpha      # Kurtosis
        self.theta = theta      # Skewness
        self.T = T              # Maturity
        self.ytm = ytm          # Yield to maturity on the debt
        self.period = period    # Coupon period in the form of decimal

        # Below holds expressions for model valuation

        self.beta = -self.theta / (self.sigma ** 2)
        self.s = self.sigma / np.sqrt(1 + ((self.theta / self.sigma) ** 2) * self.alpha / 2)
        self.c1 = (self.alpha * (self.beta + self.s) ** 2) / 2
        self.c2 = (self.alpha * self.beta ** 2) / 2

        self.ratio = (self.F * np.exp(-(self.r - self.q) * self.T)) / self.V

        self.a1 = self.k_function(self.ratio) * np.sqrt((1 - self.c1) / self.alpha)
        self.b1 = (self.beta + self.s) * np.sqrt(self.alpha / (1 - self.c1))

        self.a2 = self.k_function(self.ratio) * np.sqrt((1 - self.c2) / self.alpha)
        self.b2 = self.beta * np.sqrt(self.alpha / (1 - self.c2))

        self.y_model = self.T / self.alpha

    def k_function(self, d):
        exps = np.log(1/d) + self.T/self.alpha * np.log((1-self.c1)/(1-self.c2))

        return 1/self.s * exps

    @staticmethod
    def psi_function(a, b, y):

        # Instead of deriving psi function in terms of modified Bessel and degenerate hypergeometric functions
        # Ive used the integral expression described by Madan (1998) in the appendix.

        def integrand(t):
            left = norm.cdf(a / np.sqrt(t) + b * np.sqrt(t))
            right = ((t ** (y - 1)) * np.exp(-t)) / gamma(y)

            return left * right

        return quad(integrand, 0, np.inf)[0]

    def VGP(self):
        first_part = self.V * np.exp(-self.q * self.T) * (self.psi_function(self.a1, self.b1, self.y_model) - 1)
        second_part = -self.F * np.exp(-self.r * self.T) * (self.psi_function(self.a2, self.b2, self.y_model) - 1)

        return first_part + second_part

    def default_probability(self):
        return 1 - self.psi_function(self.a2, self.b2, self.y_model)
    
    def debt(self):
        right_side = self.V * ((self.psi_function(self.a1, self.b1, self.y_model) - 1) - self.ratio * \
                              (self.psi_function(self.a2, self.b2, self.y_model) - 1))

        return self.F * np.exp(-self.r * self.T) - right_side

    def recovery_rate(self):
        return 1 - self.VGP()/(self.F * self.default_probability() * np.exp(-self.r * self.T))

    def equity(self):
        left_side = self.V * np.exp(-self.q * self.T) * self.psi_function(self.a1, self.b1, self.y_model)
        right_side = -self.F * np.exp(-self.r * self.T) * self.psi_function(self.a2, self.b2, self.y_model)

        return left_side + right_side

    def annuity(self, i, r):
        rang = np.arange(0, self.T, i)
        sum = 0
        for i in rang:
            sum += np.exp(-i * r)
        return sum

    def credit_spread(self):
        return 10000 * self.VGP() / (self.F * self.annuity(self.period, self.ytm))

    def __repr__(self):
        abc = "For the given input parameters firm valuation is: \n" \
              "Debt = %s, Equity = %s, Probability of default = %s, Recovery rate = %s \n" \
              "and CDS spread %s BPS" % (self.debt(), self.equity(), self.default_probability(), self.recovery_rate()
                                            , self.credit_spread())
        return abc

if __name__ == "__main__":
    test = VarianceGammaCDS(2000, 1500, 0.01, 0, 0.23, 0.04, 0.13, 2, 0.012, 0.5)
    print(test)







