import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from project.task import get_model, load_data
from project.task import test as test_fn
from project.task import train as train_fn
import numpy as np

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Entrena el modelo en los datos locales."""

    # Cargar el modelo y inicializarlo con los pesos recibidos
    model_type = context.run_config.get("model-type", "mlp")
    model = get_model(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict()) # Cargar pesos del modelo (state_dict), to_torch_state_dict() convierte ArrayRecord a state_dict de PyTorch
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Leer configuración enviada por el servidor (Strategy)
    proximal_mu = msg.content["config"].get("proximal_mu", 0.0)

    # Cargar los datos
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    partition_alpha = context.run_config.get("partition-alpha", 0.1)
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, model_type, alpha=partition_alpha)

    # Llamar a la función de entrenamiento
    train_loss, train_accuracy = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        proximal_mu = proximal_mu,
    )

    # Construir y devolver el mensaje de respuesta
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model_type = context.run_config.get("model-type", "mlp")
    model = get_model(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    partition_alpha = context.run_config.get("partition-alpha", 0.1)
    _, valloader = load_data(partition_id, num_partitions, batch_size, model_type, alpha=partition_alpha)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
