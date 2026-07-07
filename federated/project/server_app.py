import os
import time
from typing import List, Tuple, Union, Optional, Dict

import matplotlib
matplotlib.use('Agg') # Optimización para entornos sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import Parameters, Scalar, FitRes
from flwr.serverapp import Grid, ServerApp
from flwr.server.client_proxy import ClientProxy
from flwr.serverapp.strategy import FedAvg, FedProx

from project.task import get_model, load_centralized_dataset, test, visualize_partitions

import toml

# Diccionario para comparar métricas entre estrategias
# Clave: Nombre de la estrategia (por ejemplo, 'FedAvg', 'FedProx')
# Valor: (lista_historial_precisión, lista_historial_pérdida)
METRIC_COMPARISON = {}
# Listas globales para almacenar temporalmente métricas del experimento ACTUAL en ejecución
CURRENT_ACCURACY_HISTORY = []
CURRENT_LOSS_HISTORY = []

CURRENT_STRATEGY_NAME = ""
CURRENT_MODEL_TYPE = ""

# Inicializar aplicación del servidor
app = ServerApp()


class TimedFedAvg(FedAvg):
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.round_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregation_result = super().aggregate_fit(server_round, results, failures)
        
        round_duration = time.time() - self.round_start_time
        print(f"\n[Round {server_round}] Duration: {round_duration:.4f} seconds")
        
        return aggregation_result

class TimedFedProx(FedProx):
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.round_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregation_result = super().aggregate_fit(server_round, results, failures)
        
        round_duration = time.time() - self.round_start_time
        print(f"\n[Round {server_round}] Duration: {round_duration:.4f} seconds")
        
        return aggregation_result


def save_run_config(context, strategy_name, model_type, mu):
    """Guarda la configuración completa del experimento en un archivo TOML."""
    
    # Cargar el pyproject original:
    with open("pyproject.toml", "r") as f:
        full_config = toml.load(f)
    
    # Construir carpeta si no existe
    os.makedirs("experiment_configs", exist_ok=True)

    # Nombre consistente
    file_name = f"{strategy_name}_{model_type}_mu{mu}.toml"
    save_path = os.path.join("experiment_configs", file_name)

    # Añadimos run_config y node_config actuales para trazar fidelidad completa
    full_config["execution"] = {
        "run_config": dict(context.run_config),
    }

    # Guardamos
    with open(save_path, "w") as f:
        toml.dump(full_config, f)

    print(f"Saved experiment config to: {save_path}")


def run_federated_experiment(grid: Grid, context: Context, strategy_name: str, strategy_class, mu: float):
    """Ejecuta un experimento completo de Aprendizaje Federado con una estrategia dada."""
    
    global CURRENT_ACCURACY_HISTORY, CURRENT_LOSS_HISTORY, CURRENT_STRATEGY_NAME, CURRENT_MODEL_TYPE
    CURRENT_STRATEGY_NAME = strategy_name
    CURRENT_MODEL_TYPE = context.run_config.get("model-type", "mlp")
    
    mu_info = f", mu={mu}" if strategy_name == "FedProx" else ""
    print(f"\n--- Starting {strategy_name} Simulation (Model: {CURRENT_MODEL_TYPE}{mu_info}) ---")

    save_run_config(context, strategy_name, CURRENT_MODEL_TYPE, mu)
    
    # Reiniciar historial para la nueva ejecución
    CURRENT_ACCURACY_HISTORY = []
    CURRENT_LOSS_HISTORY = []
    
    # Leer configuración de ejecución
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Cargar modelo global (Debe reiniciarse para cada experimento)
    global_model = get_model(CURRENT_MODEL_TYPE)
    arrays = ArrayRecord(global_model.state_dict())

    # --- Inicializar Estrategia ---
    strategy_params = {
        "fraction_evaluate": fraction_evaluate,
        "fraction_train": fraction_train,
        "min_available_nodes": context.run_config["min-available-clients"],
    }
    
    if strategy_name == 'FedProx':
        strategy = TimedFedProx(**strategy_params, proximal_mu=mu)
    else: # FedAvg (mu=0.0)
        strategy = TimedFedAvg(**strategy_params)
        
    # Iniciar simulación
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        # Pasar mu a los clientes a través de train_config
        train_config=ConfigRecord({"lr": lr, "proximal_mu": mu}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{strategy_name}_{CURRENT_MODEL_TYPE}_final_model.pt"
    save_path = os.path.join(save_dir, file_name)
    
    print(f"\nSaving final {strategy_name} model to {save_path}...")
    
    # Guardar el state_dict del resultado final
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, save_path)
    print(f"Final model successfully saved for {strategy_name}.")

    # Guardar métricas para comparación (Lógica existente)
    global METRIC_COMPARISON
    METRIC_COMPARISON[strategy_name] = (list(CURRENT_ACCURACY_HISTORY), list(CURRENT_LOSS_HISTORY))

    # --- NUEVO: Guardar la mejor ronda ---
    best_round_idx = np.argmax(CURRENT_ACCURACY_HISTORY)
    best_accuracy = CURRENT_ACCURACY_HISTORY[best_round_idx]
    best_loss = CURRENT_LOSS_HISTORY[best_round_idx]
    
    # El índice corresponde a la ronda (0 = inicial, 1 = ronda 1...)
    log_message = f"Model: {CURRENT_MODEL_TYPE}, Strategy: {strategy_name} -> Best Round: {best_round_idx}, Accuracy: {best_accuracy:.4f}, Loss: {best_loss:.4f}\n"
    print(f"\n[BEST METRICS] {log_message.strip()}")
    
    with open("best_metrics_log.txt", "a") as f:
        f.write(log_message)

    return result


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Leer configuración de ejecución
    num_rounds: int = context.run_config["num-server-rounds"]
    proximal_mu: float = context.run_config["proximal-mu"]
    strategy_config: str = context.run_config["strategy"] # Leer configuración de estrategia (por ejemplo, "fedprox")

    # 0. Visualizar distribuciones de clientes (Solo una vez)
    partition_alpha = context.run_config["partition-alpha"]
    num_clients = context.run_config["min-available-clients"]
    model_type = context.run_config.get("model-type", "mlp")
    visualize_partitions(num_partitions=num_clients, alpha=partition_alpha, model_type=model_type)

    # 1. Ejecutar FedAvg (mu=0.0)
    result_fedavg = run_federated_experiment(grid, context, 'FedAvg', FedAvg, mu=0.0)
    
    # 2.  Ejecutar FedProx (mu=0.1 desde pyproject.toml)
    result_fedprox = run_federated_experiment(grid, context, 'FedProx', FedProx, mu=proximal_mu)

    # 3. Graficar comparación de métricas
    plot_comparison_metrics(num_rounds, proximal_mu, CURRENT_MODEL_TYPE)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluar el modelo en datos centrales y almacenar el historial."""

    # Cargar el modelo e inicializarlo con los pesos recibidos
    model = get_model(CURRENT_MODEL_TYPE)
    model.load_state_dict(arrays.to_torch_state_dict())
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Cargar todo el conjunto de prueba
    test_dataloader = load_centralized_dataset()

    # Evaluar el modelo global en el conjunto de prueba
    test_loss, test_acc = test(model, test_dataloader, device)

    # Almacenar las métricas (se agregan al historial del experimento en curso)
    global CURRENT_LOSS_HISTORY, CURRENT_ACCURACY_HISTORY
    CURRENT_LOSS_HISTORY.append(test_loss)
    CURRENT_ACCURACY_HISTORY.append(test_acc)

    # Devolver las métricas de evaluación
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


def plot_comparison_metrics(num_rounds: int, proximal_mu: float, model_type: str):
    """Genera y guarda gráficos comparativos de pérdida y precisión."""
    
    # Las rondas comienzan en 0 (evaluación inicial) y llegan hasta num_rounds (11 puntos en total)
    rounds = range(0, num_rounds + 1)
    plot_dir = "metrics_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Obtener historiales
    accuracy_fedavg, loss_fedavg = METRIC_COMPARISON['FedAvg']
    accuracy_fedprox, loss_fedprox = METRIC_COMPARISON['FedProx']

    # 1. Gráfico de comparación de precisión
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracy_fedavg, marker='o', label='FedAvg')
    plt.plot(rounds, accuracy_fedprox, marker='x', label=f'FedProx (mu={proximal_mu})') # Using mu from config
    plt.title(f'Global Accuracy Comparison over Rounds ({model_type})')
    plt.xlabel('Server Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"accuracy_comparison_{model_type}.png"))
    plt.close()

    # 2. Gráfico de comparación de pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, loss_fedavg, marker='o', label='FedAvg')
    plt.plot(rounds, loss_fedprox, marker='x', label=f'FedProx (mu={proximal_mu})') # Using mu from config
    plt.title(f'Global Loss Comparison over Rounds ({model_type})')
    plt.xlabel('Server Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"loss_comparison_{model_type}.png"))
    plt.close()

    print(f"\nSaved comparison plots to the '{plot_dir}/' directory.")
    