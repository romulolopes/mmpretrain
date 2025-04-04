import os
import subprocess
import argparse

def process_images(input_base_dir, output_base_dir, config_path, checkpoint_path, method="scorecam"):
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
                
                '''
                # Comando para executar o script
                command = [
                    "python3", "tools/visualization/vis_cam.py",
                    input_path, config_path, checkpoint_path,
                    "--save-path", output_path
                ]
                '''

                # Comando para executar o script
                command = [
                    "python3", "tools/visualization/vis_cam.py",
                    input_path, config_path, checkpoint_path,
                    "--save-path", output_path,
                    "--method" , method,
                    "--aug-smooth",
                    "--eigen-smooth"
                ]
                print(f"Processando: {input_path} -> {output_path}")
                subprocess.run(command, check=True)

    print("Processamento concluído.")
    
# Chamada python view.py --type organizado
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processar múltiplas imagens com vis_cam.py, preservando a estrutura de diretórios.")
    
    parser.add_argument("--input_base_dir", help="Diretório base com as imagens de entrada.", default="")
    parser.add_argument("--output_base_dir", help="Diretório base onde os resultados serão salvos.", default="")
    parser.add_argument("--config_path", help="Caminho para o arquivo de configuração.", default="")
    parser.add_argument("--checkpoint_path", help="Caminho para o arquivo de checkpoint.", default="")
    parser.add_argument("--type", help="Tipo de processamento.",)

    args = parser.parse_args()

    if args.input_base_dir == "":
        # 'ablationcam'
        for method in ['gradcam', 'scorecam',  'xgradcam' , 'eigengradcam' , 'gradcam++']:
            for _type in ['normal', 'organizado']:
                dataset_type = _type
                input_base_dir = "data/cariotipo/test/"
                
                algorithm = "efficientnet_v2"
                config_path = f"work-dirs/{dataset_type}/{algorithm}/cariotipo.py"
                output_base_dir = f"work-dirs/{dataset_type}/output_{algorithm}_{method}/"
                checkpoint_path = f"work-dirs/{dataset_type}/{algorithm}/epoch_30.pth"
                process_images(input_base_dir, output_base_dir, config_path, checkpoint_path,method)
                
                algorithm = "mobileone"
                config_path = f"work-dirs/{dataset_type}/{algorithm}/cariotipo.py"
                output_base_dir = f"work-dirs/{dataset_type}/output_{algorithm}_{method}/"
                checkpoint_path = f"work-dirs/{dataset_type}/{algorithm}/epoch_300.pth"
                process_images(input_base_dir, output_base_dir, config_path, checkpoint_path,method)
            
                algorithm = "hrnet"
                config_path = f"work-dirs/{dataset_type}/{algorithm}/cariotipo.py"
                output_base_dir = f"work-dirs/{dataset_type}/output_{algorithm}_{method}/"
                checkpoint_path = f"work-dirs/{dataset_type}/{algorithm}/epoch_100.pth"
                process_images(input_base_dir, output_base_dir, config_path, checkpoint_path,method)
            
                algorithm = "repvgg"
                config_path = f"work-dirs/{dataset_type}/{algorithm}/cariotipo.py"
                output_base_dir = f"work-dirs/{dataset_type}/output_{algorithm}_{method}/"
                checkpoint_path = f"work-dirs/{dataset_type}/{algorithm}/epoch_120.pth"
                process_images(input_base_dir, output_base_dir, config_path, checkpoint_path,method)
            
    else:    
        process_images(args.input_base_dir, args.output_base_dir, args.config_path, args.checkpoint_path)
