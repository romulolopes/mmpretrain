import os
import subprocess
import argparse

def process_images(input_base_dir, output_base_dir, config_path, checkpoint_path):
    # Itera sobre todos os subdiretórios no diretório base de entrada
    for root, dirs, files in os.walk(input_base_dir):
        # Calcula o caminho relativo para recriar a estrutura de diretórios no output
        relative_path = os.path.relpath(root, input_base_dir)
        output_dir = os.path.join(output_base_dir, relative_path)

        # Garante que o diretório de saída exista
        os.makedirs(output_dir, exist_ok=True)

        # Processa todos os arquivos de imagem no diretório atual
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Verifica extensões de imagem
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_dir, filename)

                # Comando para executar o script
                command = [
                    "python3", "tools/visualization/vis_cam.py",
                    input_path, config_path, checkpoint_path,
                    "--save-path", output_path
                ]

                print(f"Processando: {input_path} -> {output_path}")
                subprocess.run(command, check=True)

    print("Processamento concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processar múltiplas imagens com vis_cam.py, preservando a estrutura de diretórios.")
    parser.add_argument("input_base_dir", help="Diretório base com as imagens de entrada.")
    parser.add_argument("output_base_dir", help="Diretório base onde os resultados serão salvos.")
    parser.add_argument("config_path", help="Caminho para o arquivo de configuração.")
    parser.add_argument("checkpoint_path", help="Caminho para o arquivo de checkpoint.")

    args = parser.parse_args()

    process_images(args.input_base_dir, args.output_base_dir, args.config_path, args.checkpoint_path)
