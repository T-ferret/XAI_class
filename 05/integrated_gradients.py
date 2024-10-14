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

    #'''
    integrated_grads = []

    for i, image in enumerate(x_test):
        # Integrated Gradients 계산
        integrated_grad = integrated_gradients(model, image, device=device)
        integrated_grad_np = integrated_grad.permute(1, 2, 0).detach().cpu().numpy()
        integrated_grad_np = np.sum(integrated_grad_np, axis=2)
        # [0, 1] 범위로 정규화
        integrated_grad_np = (integrated_grad_np - np.min(integrated_grad_np)) / (np.max(integrated_grad_np) - np.min(integrated_grad_np))
        integrated_grads.append(integrated_grad_np)

    integrated_grads = np.array(integrated_grads)
    visualize_gradients(org_images, pred_names, integrated_grads)
    #'''


def integrated_gradients(model, input, baseline=None, steps=100, device='cuda'):
    '''Integrated Gradients를 개선하는 함수'''
    if baseline is None:
        baseline = torch.zeros_like(input).to(device)

    # 기준 벡터에서 입력 벡터까지 선형 경로를 따라 여러 지점을 샘플링
    scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).to(device).requires_grad_(True)

    # retain_grad() 호출하여 non-leaf 텐서에 대한 기울기를 보존
    scaled_inputs.retain_grad()

    #'''
    # 모델의 출력 계산
    outputs = model(scaled_inputs)
    preds = outputs.argmax(dim=1, keepdim=True)  # 예측 클래스

    model.zero_grad()
    gradients = []

    for i in range(len(scaled_inputs)):
        outputs[i, preds[i]].backward(retain_graph=True)
        gradients.append(scaled_inputs.grad[i].clone())
        scaled_inputs.grad.zero_()  # 이전 기울기 초기화
    #'''

    gradients = torch.stack(gradients)
    avg_grads = torch.mean(gradients, dim=0)  # 기울기의 평균 계산

    integrated_grads = (input - baseline) * avg_grads  # Integrated Gradients 계산

    return integrated_grads


if __name__ == '__main__':
    main()
