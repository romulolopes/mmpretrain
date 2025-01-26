import os
import subprocess
import argparse

def process_images(input_dir, output_dir, config_path, checkpoint_path):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Verifica extensões de imagem
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            command = [
                "python3", "tools/visualization/vis_cam.py",
                input_path, config_path, checkpoint_path,
                "--save-path", output_path
            ]

            print(f"Processando: {input_path} -> {output_path}")
            subprocess.run(command, check=True)

    print("Processamento concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processar múltiplas imagens com vis_cam.py.")
    parser.add_argument("input_dir", help="Diretório com as imagens de entrada.")
    parser.add_argument("output_dir", help="Diretório onde os resultados serão salvos.")
    parser.add_argument("config_path", help="Caminho para o arquivo de configuração.")
    parser.add_argument("checkpoint_path", help="Caminho para o arquivo de checkpoint.")

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.config_path, args.checkpoint_path)
