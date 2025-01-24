import argparse
from mmpretrain import get_model, GradCAM, inference_model

def generate_grad_cam(config_path, checkpoint_path, image_path, target_layer, class_index, output_path):
    """
    Gera uma visualização Grad-CAM para um modelo treinado.

    Args:
        config_path (str): Caminho para o arquivo de configuração do modelo.
        checkpoint_path (str): Caminho para o checkpoint do modelo (.pth).
        image_path (str): Caminho para a imagem de entrada.
        target_layer (str): Nome da camada alvo para Grad-CAM.
        class_index (int): Índice da classe a ser visualizada.
        output_path (str): Caminho para salvar o resultado Grad-CAM.
    """
    # Carregar o modelo treinado
    model = get_model(config_path, checkpoint=checkpoint_path)

    # Configurar o Grad-CAM
    grad_cam = GradCAM(model, target_layer_name=target_layer)

    # Fazer inferência na imagem
    result = inference_model(model, image_path)

    # Gerar o Grad-CAM
    cam_image = grad_cam(result, target_category=class_index)

    # Salvar o Grad-CAM
    grad_cam.show_cam_on_image(image_path, cam_image, save_path=output_path)
    print(f"Grad-CAM salvo em: {output_path}")

if __name__ == "__main__":
    # Configurar o parser de argumentos
    parser = argparse.ArgumentParser(description="Geração de Grad-CAM para modelos treinados no MMPretrain.")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo de configuração do modelo (.py).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho para o checkpoint do modelo (.pth).")
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem de entrada.")
    parser.add_argument("--layer", type=str, required=True, help="Nome da camada alvo para Grad-CAM.")
    parser.add_argument("--class_index", type=int, required=True, help="Índice da classe a ser visualizada.")
    parser.add_argument("--output", type=str, required=True, help="Caminho para salvar o resultado Grad-CAM.")

    # Parse dos argumentos
    args = parser.parse_args()

    # Gerar Grad-CAM com os parâmetros fornecidos
    generate_grad_cam(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        target_layer=args.layer,
        class_index=args.class_index,
        output_path=args.output
    )
