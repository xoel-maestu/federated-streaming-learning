import os
import random
from collections import Counter
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner


FDS = None  # Cache FederatedDataset

# Normalizamos los datos de Fashion-MNIST a partir de sus estadísticas, para mejorar el rendimiento del entrenamiento
FASHION_MEAN = 0.2860
FASHION_STD = 0.3530
pytorch_transforms = Compose(
    [
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((FASHION_MEAN,), (FASHION_STD,)),
    ]
)


# Definimos un MLP simple
class Net(nn.Module):
    """
    Red neuronal simple tipo MLP (Multi-Layer Perceptron).
    Entrada: Imágenes aplanadas de 28x28 (784 features).
    Salida: 10 clases (Fashion-MNIST).
    """
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()  # Aplana la imagen de 28x28 a un vector de 784
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LayerNorm(512), # Normalización de capa para estabilidad
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256), # Normalización de capa para estabilidad
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)  # Aplana la entrada
        logits = self.linear_relu_stack(x)  # Pasa la entrada a través de la red
        return logits

class CNN(nn.Module):
    """
    Red neuronal convolucional (CNN) optimizada.
    Simplificada para eficiencia y mejor rendimiento en Fashion-MNIST.
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Bloque Convolucional 1: 32 filtros, kernel 3x3 (estándar)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bloque Convolucional 2: 64 filtros, kernel 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        
        # Capas Totalmente Conectadas
        # Entrada: 64 filtros * 7 * 7 (después de 2 pooling 28->14->7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Regularización
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Aplanar
        x = torch.flatten(x, 1)
        
        # FC 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.fc2(x)
        return x




def replace_bn_with_gn(module, num_groups=8):
    """
    Reemplaza recursivamente BatchNorm2d con GroupNorm.
    Ajusta el número de grupos si el número de canales no es divisible por num_groups.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            
            # Buscar un número de grupos válido (que divida a num_channels)
            # Empezamos con el deseado (8) y vamos bajando si no divide exacto.
            groups = num_groups
            while groups > 1 and num_channels % groups != 0:
                groups //= 2
            
            # Reemplazar la capa
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            replace_bn_with_gn(child, num_groups)


def get_model(model_type: str = "mlp"):
    if model_type.lower() == "cnn":
        return CNN()
    elif model_type.lower() == "mlp":
        return Net()
    
    elif model_type.lower() == "resnet":
        # 1. Cargar ResNet18 pre-entrenada
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Adaptar la primera capa para Fashion-MNIST (1 canal, 28x28)
        # Cambiamos stride=1 para no reducir la imagen demasiado rápido
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 3. Eliminar MaxPool inicial para preservar resolución espacial
        model.maxpool = nn.Identity()

        # 4. Modificar la capa de salida para 10 clases
        model.fc = nn.Linear(model.fc.in_features, 10)
        
        # 5. Reemplazar BatchNorm con GroupNorm para estabilidad en FL Non-IID
        replace_bn_with_gn(model)
        
        return model

    elif model_type.lower() == "mobilenet":
        # 1. Cargar MobileNetV3 Small
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # 2. Adaptar primera capa (1 canal, stride=1)
        # MobileNet empieza con una Conv2d dentro de 'features'
        model.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 3. Modificar el clasificador final para 10 clases
        # La estructura es: Sequential(Linear, Hardswish, Dropout, Linear)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)

        # 4. ¡CRÍTICO! Reemplazar BatchNorm con GroupNorm
        replace_bn_with_gn(model)

        return model
    else:
        raise ValueError(f"Model type {model_type} not supported")


def apply_transforms(batch):
    """Aplica las transformaciones definidas a un batch de datos."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def get_target_labels(partition_id: int, partition) -> set:
    """
    Selecciona las etiquetas a mantener para un cliente específico.
    Si la partición está balanceada (alpha alto), esto selecciona aleatoriamente
    2 o 3 clases debido a las fluctuaciones naturales o se podría forzar aleatoriedad.
    Aquí confiamos en el conteo (most_common) que funciona tanto para alpha bajo como alto.
    """
    rng = random.Random(partition_id)
    k_classes = rng.choice([2, 3])
    
    all_labels = partition["label"]
    label_counts = Counter(all_labels)
    
    most_common = label_counts.most_common(k_classes)
    kept_labels = {label for label, count in most_common}
    return kept_labels



def get_partitioned_dataset(num_partitions: int, alpha: float, model_type: str = "mlp"):
    """
    Helper function to get the FederatedDataset with the correct partitioner
    based on the model type.
    
    RESNET/MOBILENET -> IID
    MLP/CNN -> Non-IID (Dirichlet)
    """
    global FDS
    
    if FDS is not None:
        return FDS

    if model_type.lower() in ("resnet", "mobilenet"):
        print(f"Initializing IID Partitioner for model {model_type}")
        partitioner = IidPartitioner(
            num_partitions=num_partitions,
        )
    else:
        print(f"Initializing Dirichlet Partitioner (alpha={alpha}) for model {model_type}")
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,
            min_partition_size=10,
            seed=42,
        )
    
    FDS = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": partitioner},
    )
    return FDS


def load_data(
    partition_id: int, num_partitions: int, batch_size: int, model_type : str, alpha: float = 0.1
):
    """Carga los datos para un cliente específico basado en la partición."""
    
    # Obtener dataset compartido (asegura misma inicialización)
    fds = get_partitioned_dataset(num_partitions, alpha, model_type)
    
    partition = fds.load_partition(partition_id)

    # Solo filtrar clases si NO es IID (MLP/CNN)
    if model_type.lower() not in ("resnet", "mobilenet"):
        # Filtrar para mantener solo las K clases
        kept_labels = get_target_labels(partition_id, partition)
        # Filtrar el conjunto de datos
        partition = partition.filter(lambda x: x["label"] in kept_labels)

    # Configurar el formato de los datos para PyTorch
    partition.set_format("torch")

    # Dividir en train/test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Crear DataLoaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Carga el conjunto de datos de prueba centralizado."""
    test_dataset = load_dataset("zalando-datasets/fashion_mnist", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device, proximal_mu=0.0):
    # Guardar copia de parámetros globales para FedProx
    global_params = [p.clone().detach() for p in net.parameters()]
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            # termino proximal de FedProx
            if proximal_mu > 0.0:
                proximal_term = 0.0
                for local_weights, global_weights in zip(
                    net.parameters(), global_params
                ):
                    proximal_term += (local_weights - global_weights).norm(2) ** 2
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Calcular precisión de entrenamiento
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    train_accuracy = correct / total
    return avg_trainloss, train_accuracy


def test(net, testloader, device):
    """Validar el modelo en el conjunto de prueba."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def visualize_partitions(num_partitions: int, alpha: float, model_type: str = "mlp"):
    """Generates a single figure with histograms for all client partitions."""
    print(f"Generating consolidated partition visualization (alpha={alpha}, model={model_type})...")
    
    # Use shared helper to get the correct partitioner (IID or Dirichlet)
    # This ensures visualization matches the actual data distribution used in training
    fds = get_partitioned_dataset(num_partitions, alpha, model_type)
    
    save_dir = "client_distributions"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"client_distributions_{model_type}.png")

    # Always regenerate
    if os.path.exists(save_path):
        print(f"Visualization already exists at {save_path}. Overwriting...")

    cols = 2
    rows = (num_partitions + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i in range(num_partitions):
        partition = fds.load_partition(i, "train")
        all_labels = partition["label"]
        
        # Only filter for Non-IID models (MLP, CNN) that use the "keep K classes" logic
        # For IID models (ResNet, MobileNet), we show the full distribution
        if model_type.lower() not in ("resnet", "mobilenet"):
            kept_labels = get_target_labels(i, partition)
            labels = [l for l in all_labels if l in kept_labels]
        else:
            # IID: Keep all labels
            labels = all_labels

        counts = np.bincount(labels, minlength=10)
        
        ax = axes[i]
        ax.bar(range(10), counts)
        ax.set_title(f"Client {i} ({'IID' if model_type.lower() in ('resnet', 'mobilenet') else 'Dirichlet'})")
        ax.set_xticks(range(10))
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        
    # Hide empty subplots
    for j in range(num_partitions, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved consolidated client distributions to {save_path}")
