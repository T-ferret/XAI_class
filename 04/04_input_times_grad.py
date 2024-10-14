import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from utils import *


def main():

    test_dir = '../../test_data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()

    # 모델 구조 출력
    #print(model)

    preprocess = weights.transforms()
    mean = np.array(preprocess.mean).reshape(1, 3, 1, 1)
    std = np.array(preprocess.std).reshape(1, 3, 1, 1)

    x_test, y_test = get_test_imgs(test_dir, preprocess)
    x_test = x_test.to(device)
    org_images = denomrmalize(x_test.cpu(), mean, std)
    x_test.requires_grad = True

    # 정답 출력
    print(y_test)

    # 모델 예측
    y_pred = model(x_test).softmax(1)
    pred_id = y_pred.argmax(1).cpu().numpy()

    # 예측 라벨 출력
    print(pred_id)
    pred_names = get_class_name(weights, pred_id)
    # plot_images(org_images, pred_names)

    # 모델 출력에 대한 입력의 기울기 계산
    model.zero_grad()
    for i in range(len(x_test)):
        output = model(x_test)
        pred = output.argmax(1)
        output[i, pred[i]].backward(retain_graph=True)

    # 기울기 가져오기
    gradients = x_test.grad.data

    # Input*Gradient 계산
    input_gradients = x_test * gradients
    input_gradients_np = input_gradients.permute(0, 2, 3, 1).detach().cpu().numpy()

    # Normalize to [0, 1]
    input_gradient_np = (input_gradients_np-np.min(input_gradients_np))/(np.max(input_gradients_np)-np.min(input_gradients_np))
    visualize_gradients(org_images, pred_names, input_gradient_np)


if __name__ == '__main__':
    main()
