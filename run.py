# -*- coding: utf-8 -*-
import numpy as np
import pulp as pl
import os
from communication_strategy.communication_strategy import Communication_Strategy
from server.server import Server
from transmission_model.transmission_model import Transmission_Model


class CS(Communication_Strategy):

    def __init__(self, transmission_model, min_fit_clients, clients_number_data_samples,
                 delay_requirement, energy_requirement):

        self.clients_number_data_samples = clients_number_data_samples
        super().__init__(transmission_model, min_fit_clients, delay_requirement, energy_requirement)

    def greater_data_user_selection(self, factor, k):
        print(f"> greater_data")

        selected_clients = np.random.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        print(f"user_selection: {selected_clients}")

        data_samples_list = np.array(self.clients_number_data_samples)[selected_clients]
        pos_list = np.arange(len(data_samples_list))
        print(data_samples_list)

        # Combinação das listas
        combined_data = list(zip(data_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        distance_list, pos_list = zip(*sorted_data)

        print(f"data_list: {distance_list}")
        print(f"pos_list: {pos_list}")

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])
        print(f"final user_selection: {final_selected_clients}")

        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[final_selected_clients] = 1

    def greater_loss_user_selection(self, clients_loss_list, factor, k):
        print(f"> greater_LOSS")

        selected_clients = np.random.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        print(f"user_selection: {selected_clients}")

        loss_samples_list = np.array(clients_loss_list)[selected_clients]
        pos_list = np.arange(len(loss_samples_list))
        print(loss_samples_list)

        # Combination of lists
        combined_data = list(zip(loss_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        loss_list, pos_list = zip(*sorted_data)

        print(f"data_list: {loss_list}")
        print(f"pos_list: {pos_list}")

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])
        print(f"final user_selection: {final_selected_clients}")

        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[final_selected_clients] = 1

    def users_closer_user_selection(self, factor, k):
        print(f"> @@@@@@@@ users_closer_user_selection")

        selected_clients = np.random.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        print(f"user_selection: {selected_clients}")

        data_distance_list = np.array(self.tm.user_distance)[selected_clients]
        pos_list = np.arange(len(data_distance_list))

        # Combination of lists
        combined_data = list(zip(data_distance_list, pos_list))
        sorted_data = sorted(combined_data, reverse=False, key=lambda x: x[0])
        distance_list, pos_list = zip(*sorted_data)

        print(f"data_list: {distance_list}")
        print(f"pos_list: {pos_list}")

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])
        print(f"final user_selection: {final_selected_clients}")

        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[final_selected_clients] = 1

    def random_user_selection(self):
        print(f">> random_user_selection")
        super().random_user_selection()

    def random_rb_allocation(self):
        print(f">> random_rb_allocation")
        super().random_rb_allocation()

    def optimization_rb_allocation(self):
        print(f"> optimization_rb_allocation")
        selected_clients = np.where(self.selected_clients > 0)[0].tolist()
        print(f"selected_clients: {selected_clients}")

        prob_list = [self.W[i].tolist() for i in selected_clients]

        # Data
        number_clients = len(prob_list)  # num_tarefas
        number_channels = len(prob_list[0])  # num_agentes
        print(number_clients)
        print(number_channels)

        # Creation of the assignment problem
        model = pl.LpProblem("Max_Prob", pl.LpMaximize)

        # Decision Variables
        x = [[pl.LpVariable(f"x_{i}_{j}", cat=pl.LpBinary) for j in range(number_channels)] for i in
             range(number_clients)]

        # Objective function
        model += pl.lpSum(prob_list[i][j] * x[i][j] for i in range(number_clients) for j in
                          range(number_channels)), "Custo_Prob_Total"

        # Constraints: Each customer is assigned to exactly one channel
        for i in range(number_clients):
            model += pl.lpSum(x[i][j] for j in range(number_channels)) >= 0, f"Restricao_Cliente_Canal_{i} >= 0"

        for i in range(number_clients):
            model += pl.lpSum(x[i][j] for j in range(number_channels)) <= 1, f"Restricao_Cliente_Canal_{i} <= 1"

        # Constraints: Each channel is assigned to exactly one customer
        for j in range(number_channels):
            model += pl.lpSum(x[i][j] for i in range(number_clients)) >= 0, f"Restricao_Canal_Cliente_{j} >= 0"

        for j in range(number_channels):
            model += pl.lpSum(x[i][j] for i in range(number_clients)) <= 1, f"Restricao_Canal_Cliente_{j} <= 1"

        model += pl.lpSum([x[i][j] for i in range(number_clients) for j in
                           range(number_channels)]) == self.min_fit_clients, f"Clientes selecionados"

        # Solving the problem
        model.solve()

        allocated_channels = []
        update_selected_clients = []
        for i in range(number_clients):
            for j in range(number_channels):
                if pl.value(x[i][j]) == 1:
                    allocated_channels.append(j)
                    update_selected_clients.append(selected_clients[i])

        print(f">>>>>> *: old: selected_clients: {selected_clients}")
        print(f">>>>>> *: update_selected_clients: {update_selected_clients}")
        print(f">>>>>> *: allocated_channels: {allocated_channels}")

        # Update selected
        self.rb_allocation = np.zeros(self.tm.user_number, dtype=int)
        for ind, channel in enumerate(allocated_channels):
            self.rb_allocation[update_selected_clients[ind]] = channel

        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[update_selected_clients] = 1
        print("<<<<<<<<<")

    # Better channels for more distant customers
    def rb_allocation_baseline_x(self):

        print(f"> rb_allocation_baseline_x")
        selected_clients = np.where(self.selected_clients > 0)[0].tolist()
        print(f"selected_clients: {selected_clients}")

        distance_list = [x[0] for x in np.array(self.tm.user_distance)[selected_clients]]
        pos_list = list(range(len(distance_list)))

        print(distance_list)
        combined_data = list(zip(distance_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        distance_list, pos_list = zip(*sorted_data)

        allocated_channels = np.array([-1] * self.tm.rb_number)
        for i in pos_list:
            cli = selected_clients[i]
            packet_error_list = self.tm.q[cli].copy()
            packet_error_list[allocated_channels[np.where(np.array(allocated_channels) > -1)[0].tolist()]] = np.inf
            ind = np.argmin(packet_error_list)
            allocated_channels[i] = ind

        print(allocated_channels)

        self.rb_allocation = np.zeros(self.tm.user_number, dtype=int)
        for ind, channel in enumerate(allocated_channels):
            if channel >= 0:
                self.rb_allocation[selected_clients[ind]] = channel

        print(self.rb_allocation)


class FL(Server):

    def __init__(self, n_rounds, total_number_clients, min_fit_clients, rb_number, load_client_data_constructor,
                 path_server, path_clients, shape, model_type, parallel_processing=False):
        super().__init__(n_rounds, total_number_clients, min_fit_clients, load_client_data_constructor,
                         path_server, path_clients, shape, model_type, parallel_processing)

        self.strategy = CS(
            Transmission_Model(rb_number=rb_number, user_number=total_number_clients,
                               total_model_params=self.model.count_params(),
                               lower_limit_distance=100, upper_limit_distance=500),
            min_fit_clients=min_fit_clients, clients_number_data_samples=self.clients_number_data_samples,
            delay_requirement=0.2, energy_requirement=0.0025)
        # delay_requirement=0.2, energy_requirement=0.0025 - NIID R-MNIST com MLP
        # delay_requirement=0.4, energy_requirement=0.005  - NIID R-FMNIST com CNN

        self.evaluate_list = {"distributed": {"loss": [], "accuracy": []}, "centralized": {"loss": [], "accuracy": []}}
        self.success_uploads = []
        self.error_uploads = []

        self.delay_success_sum_uploads = []
        self.delay_error_sum_uploads = []

        self.delay_success_max_uploads = []
        self.delay_max = []

        self.energy_success_sum_uploads = []
        self.energy_error_sum_uploads = []

    def configure_fit(self):
        # FedAvg
        self.strategy.random_user_selection()
        self.strategy.random_rb_allocation()

        # POC
        # self.strategy.greater_loss_user_selection(clients_loss_list=fl.clients_loss, factor=2, k=self.min_fit_clients)
        # self.strategy.random_rb_allocation()

        # FedAvg-wOpt
        # self.strategy.random_user_selection()
        # self.strategy.optimization_rb_allocation()

        # POC-wOpt
        # self.strategy.greater_loss_user_selection(clients_loss_list=fl.clients_loss, factor=2, k=self.min_fit_clients)
        # self.strategy.optimization_rb_allocation()

        # DFed-wOpt
        # self.strategy.greater_data_user_selection(factor=2, k=self.strategy.tm.rb_number)
        # self.strategy.optimization_rb_allocation()

        ################
        self.selected_clients = np.where(self.strategy.selected_clients > 0)[0].tolist()
        self.strategy.compute_final_values()


if __name__ == "__main__":

    os.system('clear')

    fl = FL(n_rounds=200,
            min_fit_clients=10,
            rb_number=15,
            total_number_clients=100,
            path_server="../datasets/mnist/mnist",
            path_clients="../datasets/mnist/non-iid-0.9-100-rotation-45",
            # path_server="../datasets/fashion-mnist/fashion-mnist",
            # path_clients="../datasets/fashion-mnist/non-iid-0.9-100-rotation",
            shape=(28, 28, 1),
            model_type="MLP",
            load_client_data_constructor=False)

    print(f"INÍCIO")
    fl.strategy.print_total()
    print("clients_number_data_samples")
    print(fl.clients_number_data_samples)
    print("Distancia")
    print(fl.strategy.tm.user_distance)

    evaluate_loss, evaluate_accuracy = None, None
    # Communication rounds
    for fl.server_round in range(fl.n_rounds):
        # Select customers who will participate in the next round of communication
        fl.configure_fit()

        fl.strategy.print_final_values()

        success_uploads, error_uploads = fl.strategy.list_upload_status()
        print(f"success_uploads: {success_uploads} - error_uploads: {error_uploads}")
        fl.success_uploads.append(len(success_uploads))
        fl.error_uploads.append(len(error_uploads))

        (delay_success_max_uploads, delay_max,
         delay_success_sum, delay_error_sum,
         energy_success_sum, energy_error_sum) = fl.strategy.round_costs(success_uploads, error_uploads)

        fl.delay_success_max_uploads.append(delay_success_max_uploads)
        fl.delay_max.append(delay_max)
        fl.delay_success_sum_uploads.append(delay_success_sum)
        fl.delay_error_sum_uploads.append(delay_error_sum)
        fl.energy_success_sum_uploads.append(energy_success_sum)
        fl.energy_error_sum_uploads.append(energy_error_sum)

        if len(success_uploads) > 0:
            # On-device training

            for cid in fl.selected_clients:
                fl.count_of_client_uploads[cid] = fl.count_of_client_uploads[cid] + 1

            fl.selected_clients = np.array(fl.selected_clients)[success_uploads]
            weight_list, sample_sizes, info = fl.fit()
            print("ACC List")
            print(fl.clients_acc)
            print("Loss List")
            print(fl.clients_loss)
            print("Qtde Upload client")
            print(fl.count_of_client_uploads)

            # Aggregation
            fl.aggregate_fit(weight_list, sample_sizes)

            print(f"***************************")
            # Centralized evaluate
            print(f"Centralized evaluate: R: {fl.server_round + 1} ")
            evaluate_loss, evaluate_accuracy = fl.centralized_evaluation()
            print(f"evaluate_accuracy: {evaluate_accuracy}")

        fl.evaluate_list["centralized"]["loss"].append(evaluate_loss)
        fl.evaluate_list["centralized"]["accuracy"].append(evaluate_accuracy)

        print(f"-----------------")

    print(f"\ncentralized_accuracy: ")
    print(fl.evaluate_list["centralized"]["accuracy"])

    print(f"\ncentralized_loss: ")
    print(fl.evaluate_list["centralized"]["loss"])

    print(f"\nsuccess_uploads:")
    print(fl.success_uploads)
    print(np.cumsum(fl.success_uploads).tolist())

    print(f"\nerror_uploads:")
    print(fl.error_uploads)
    print(np.cumsum(fl.error_uploads).tolist())

    print(f"\ndelay_success_sum_uploads:")
    print(fl.delay_success_sum_uploads)
    print(np.cumsum(fl.delay_success_sum_uploads).tolist())
    print(f"\ndelay_error_sum_uploads:")
    print(fl.delay_error_sum_uploads)
    print(np.cumsum(fl.delay_error_sum_uploads).tolist())

    print(f"\ndelay_success_max_uploads:")
    print(fl.delay_success_max_uploads)
    print(np.cumsum(fl.delay_success_max_uploads).tolist())
    print(f"\ndelay_max:")
    print(fl.delay_max)
    print(np.cumsum(fl.delay_max).tolist())

    print(f"\nenergy_success_sum_uploads:")
    print(fl.energy_success_sum_uploads)
    print(np.cumsum(fl.energy_success_sum_uploads).tolist())
    print(f"\nenergy_error_sum_uploads:")
    print(fl.energy_error_sum_uploads)
    print(np.cumsum(fl.energy_error_sum_uploads).tolist())

    print("\nQdde Upload client")
    print(fl.count_of_client_uploads)
    print(f"\nFIM")
