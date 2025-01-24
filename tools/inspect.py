import argparse
import sys
from pathlib import Path

# Adicionar o diretório raiz do projeto ao sys.path
project_root = Path(__file__).resolve().parents[1]  # Caminho do diretório "mmpretrain"
sys.path.append(str(project_root))

# Importar o módulo do projeto
from mmpretrain import get_model

def inspect_model(config_path, checkpoint_path):
    """
    Carrega e exibe a estrutura do modelo baseado no MMPretrain.

    Args:
        config_path (str): Caminho para o arquivo de configuração do modelo (.py).
        checkpoint_path (str): Caminho para o checkpoint do modelo (.pth).
    """
    # Carregar o modelo
    model = get_model(config_path, checkpoint=checkpoint_path)

    # Inspecionar a estrutura do modelo
    print("\nEstrutura do modelo:\n")
    print(model)

if __name__ == "__main__":
    # Configurar argparse
    parser = argparse.ArgumentParser(description="Inspecionar a estrutura de um modelo do MMPretrain.")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo de configuração do modelo (.py).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho para o checkpoint do modelo (.pth).")

    # Parse dos argumentos
    args = parser.parse_args()

    # Chamar a função para inspecionar o modelo
    inspect_model(config_path=args.config, checkpoint_path=args.checkpoint)
