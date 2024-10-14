from torchvision.models import resnet50, ResNet50_Weights
from utils import *
import matplotlib.pyplot as plt


# 활성화 맵 시각화 함수
def visualize_activation_maps(activation, file_path):
    fig, ax_arr = plt.subplots(activation.shape[1] // 8, 8, figsize=(15, 15))

    for idx in range(activation.shape[1]):
        ax_arr[idx // 8, idx % 8].imshow(activation[0][idx].cpu().detach().numpy(), cmap='jet')
        ax_arr[idx // 8, idx % 8].axis('off')
    plt.show()
    fig.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# ResNet50 모델의 특정 레이어 활성화 맵 열기
activation = None


def get_activation(name):
    def hook(model, input, output):
        global activation
        activation = output
    return hook


def main():
    test_dir = "../../test_data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()

    # 모델 구조 출력
    # print(model)

    preprocess = weights.transforms()
    mean = np.array(preprocess.mean).reshape(1, 3, 1, 1)
    std = np.array(preprocess.std).reshape(1, 3, 1, 1)

    X_test, y_test = get_test_imgs(test_dir, preprocess)
    X_test = X_test.to(device)
    org_imgs = denomrmalize(X_test.cpu(), mean, std)
    X_test.requires_grad = True

    # 정답 출력
    print(y_test)

    # 모델 예측
    y_pred = model(X_test).softmax(1)
    pred_id = y_pred.argmax(1).cpu().numpy()

    # 예측 레이블 출력
    print(pred_id)
    pred_names = get_class_name(weights, pred_id)
    plot_images(org_imgs, pred_names)

    target_layer = model.layer4[0].conv1
    target_layer_name = 'layer4[0].conv1'
    hook_handle = target_layer.register_forward_hook(get_activation(target_layer_name))

    # 모델 통과
    output = model(X_test[:1])

    # 활성화 맵 시각화 및 저장
    file_path = '.\\layer_visualization_map\\%s.png' % target_layer_name
    visualize_activation_maps(activation, file_path)

    # 훅 제거
    hook_handle.remove()


if __name__ == '__main__':
    main()