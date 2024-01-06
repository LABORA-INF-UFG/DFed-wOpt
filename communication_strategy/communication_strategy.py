import numpy as np

# np.random.seed(1)


class Communication_Strategy:

    def __init__(self, transmission_model, min_fit_clients, delay_requirement=0.5, energy_requirement=0.003):
        self.tm = transmission_model
        self.min_fit_clients = min_fit_clients
        self.delay_requirement = delay_requirement
        self.energy_requirement = energy_requirement

        self.selected_clients = np.array([])
        self.rb_allocation = np.array([])

        self.final_total_delay = np.array([])
        self.final_total_energy = np.array([])
        self.W = np.array([])
        self.final_W = np.array([])
        self.init()

    def init(self):
        self.compute_transmission_probability_matrix()

    def compute_final_values(self):
        self.compute_final_total_delay()
        self.compute_final_total_energy()
        self.compute_final_w()

    def print_final_values(self):
        print(f"selected_clients: {np.where(self.selected_clients > 0)[0] + 1}")
        print(f"rb_allocation: {self.rb_allocation[np.where(self.selected_clients > 0)]}")
        print(f"Total Delay | Total Energy | W | Distance")
        print(self.final_total_delay[np.where(self.selected_clients > 0)])
        print(self.final_total_energy[np.where(self.selected_clients > 0)])
        print(self.final_W[np.where(self.selected_clients > 0)])
        print(self.tm.user_distance[np.where(self.selected_clients > 0)])

    def random_user_selection(self):
        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[np.random.permutation(self.tm.user_number)[:self.min_fit_clients]] = 1

    def random_rb_allocation(self):
        self.rb_allocation = np.zeros(self.tm.user_number, dtype=int)
        self.rb_allocation[np.where(self.selected_clients > 0)] = np.random.permutation(self.tm.rb_number)[
                                                                  :self.min_fit_clients]

    def compute_final_total_delay(self):
        self.final_total_delay = np.zeros(self.tm.user_number, dtype=float)
        for i in np.where(self.selected_clients > 0)[0].tolist():
            self.final_total_delay[i] = self.tm.total_delay[i, self.rb_allocation[i]]

    def compute_final_total_energy(self):
        self.final_total_energy = np.zeros(self.tm.user_number, dtype=float)
        for i in np.where(self.selected_clients > 0)[0].tolist():
            self.final_total_energy[i] = self.tm.total_energy[i, self.rb_allocation[i]]

    def compute_final_w(self):
        self.final_W = np.zeros(self.tm.user_number, dtype=float)
        for i in np.where(self.selected_clients > 0)[0].tolist():
            self.final_W[i] = self.W[i, self.rb_allocation[i]]

    def compute_transmission_probability_matrix(self):
        self.W = np.zeros((self.tm.user_number, self.tm.rb_number))
        for i in range(self.tm.user_number):
            for j in range(self.tm.rb_number):
                if self.tm.total_delay[i, j] < self.delay_requirement and self.tm.total_energy[
                    i, j] < self.energy_requirement:
                    self.W[i, j] = 1 - self.tm.q[i, j]

    def print_total(self):
        print(f"==> total_delay")
        for i in range(self.tm.user_number):
            print(
                f"---> {[i]} > {self.tm.user_distance[i]} \n > D: {self.tm.total_delay[i]} \n > E: {self.tm.total_energy[i]} \n > Prob: {self.W[i]}")

    def list_upload_status(self):
        prob = np.random.rand(self.min_fit_clients)
        print(f"Error Prob: {prob}")

        values = self.final_W[np.where(self.selected_clients > 0)]
        success_uploads = np.where((values > 0) & (values >= prob))[0]
        error_uploads = np.delete(np.arange(self.min_fit_clients), success_uploads)

        return success_uploads, error_uploads

    def round_costs(self, success_uploads, error_uploads):
        delay_success_list = self.final_total_delay[np.where(self.selected_clients > 0)][success_uploads]
        delay_error_list = self.final_total_delay[np.where(self.selected_clients > 0)][error_uploads]
        energy_success_list = self.final_total_energy[np.where(self.selected_clients > 0)][success_uploads]
        energy_error_list = self.final_total_energy[np.where(self.selected_clients > 0)][error_uploads]

        print(f"delay_success_uploads: {delay_success_list}")
        print(f"delay_error_uploads: {delay_error_list}")
        print(f"energy_success_uploads: {energy_success_list}")
        print(f"energy_error_uploads: {energy_error_list}")

        delay_success_max_uploads = max(delay_success_list) if len(delay_success_list) > 0 else 0
        delay_max = max([delay_success_max_uploads, max(delay_error_list) if len(delay_error_list) > 0 else 0])

        delay_success_sum = sum(delay_success_list)
        delay_error_sum = sum(delay_error_list)
        energy_success_sum = sum(energy_success_list)
        energy_error_sum = sum(energy_error_list)

        return delay_success_max_uploads, delay_max, delay_success_sum, delay_error_sum, energy_success_sum, energy_error_sum
