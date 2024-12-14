import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

YSV = {1: 0.03225806451612903, 2: 0.034482758620689655, 3: 0.03225806451612903,
                       4: 0.03333333333333333, 5: 0.03225806451612903, 6: 0.03333333333333333,
                       7: 0.03225806451612903, 8: 0.03225806451612903, 9: 0.03333333333333333,
                       10: 0.03225806451612903, 11: 0.03333333333333333, 12: 0.03225806451612903}


def transform_u(u, a, b):
    return a + ((b - a) * u)



class Demand2:

    def __init__(self, params, scaling_factor):

        self.params = params
        self.dr = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        self.scales = scaling_factor


    def sim_day(self, month):
        a, b = self.params.loc[month]
        monthly_cons = np.random.uniform(a, b)
        est = monthly_cons * YSV[month] * self.scales.loc[month]

        return est


    def simulate_year(self):
        path = []
        for date in self.dr:
            mon = date.month
            est = self.sim_day(mon)
            path.append(est)
        return np.array(path)


    def sim_day_antithetic(self, u, month):
        a, b = self.params.loc[month]
        monthly_cons = transform_u(u, a, b)
        est = monthly_cons * YSV[month] * self.scales.loc[month]

        return est


    def simulate_year_antithetic(self):
        path = []

        U0 = np.random.rand(182)
        U = np.ravel(np.column_stack((U0, 1 - U0)))
        U = np.append(U, np.random.rand())

        for i in range(len(self.dr)):
            date = self.dr[i]
            mon = date.month
            est = self.sim_day_antithetic(U[i], mon)
            path.append(est)

        return np.array(path)




class UniversalGenerator:

    def __init__(self, params, ic, P, reduction):
        self.params = params
        self.dr = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        self.ic = ic
        self.P = P
        self.reduction = reduction


    def sim_day(self, month):
        a, b = self.params.loc[month]
        monthly_gen = np.random.uniform(a, b)
        est = monthly_gen * YSV[month] * self.ic

        return est

    @staticmethod
    def _transform_state(states, reduction):
        new_states = [reduction if states[i]==1 else 1 for i in range(len(states))]
        # return np.where(states == 1, reduction, states)
        return np.array(new_states)


    def simulate_year(self):
        path = []
        initial_state = 0
        states = np.zeros(365, dtype=int)
        states[1] = initial_state

        # Simulate the Markov chain
        for day in range(1, 365):
            current_state = states[day - 1]
            U = np.random.random()
            if U < self.P[current_state, 0]:
                states[day] = 0
            else:
                states[day] = 1

        for date in self.dr:
            mon = date.month
            est = self.sim_day(mon)
            path.append(est)

        t_states = self._transform_state(states, self.reduction)

        path2 = path * t_states

        self.states = states
        return np.array(path2)



    def _disruption_DTMC(self):
        initial_state = 0
        states = np.zeros(365, dtype=int)
        states[1] = initial_state

        # Simulate the Markov chain
        for day in range(1, 365):
            current_state = states[day - 1]
            U = np.random.random()
            if U < self.P[current_state, 0]:
                states[day] = 0
            else:
                states[day] = 1

        t_states = self._transform_state(states, self.reduction)
        return t_states



    def sim_day_anthithetic(self, u, month):
        a, b = self.params.loc[month]
        monthly_gen = transform_u(u, a, b)
        est = monthly_gen * YSV[month] * self.ic

        return est

    def simulate_year_anthithetic(self):
        path = []

        U0 = np.random.rand(182)
        U = np.ravel(np.column_stack((U0, 1 - U0)))
        U = np.append(U, np.random.rand())

        for i in range(len(self.dr)):

            date = self.dr[i]
            mon = date.month
            est = self.sim_day_anthithetic(U[i], mon)
            path.append(est)


        t_states = self._disruption_DTMC()
        path2 = path * t_states

        return np.array(path2)

