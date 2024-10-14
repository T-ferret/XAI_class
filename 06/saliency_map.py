from torchvision.models import resnet50, ResNet50_Weights
from utils import *
import numpy as np


def get_saliency_map(model, input_images):

    input_images.requires_grad = True
    output = model(input_images)
    pred_classes = torch.argmax(output, dim=1)

    model.zero_grad()
    output.backward(torch.ones_like(output))

    saliencies = []

    for i in range(pred_classes.size(0)):
        saliency, _ = torch.max(input_images.grad[i].abs(), dim=0)
        saliencies.append(saliency.cpu().numpy())

    return saliencies


def main():

    test_dir = '../../test_data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    mean = np.array(preprocess.mean).reshape(1, 3, 1, 1)
    std = np.array(preprocess.std).reshape(1, 3, 1, 1)

    x_test, y_test = get_test_imgs(test_dir, preprocess)
    x_test = x_test.to(device)
    org_images = denomrmalize(x_test.cpu(), mean=mean, std=std)
    print(y_test)

    y_pred = model(x_test).softmax(0)
    pred_id = y_pred.argmax(1)
    print(pred_id.cpu().numpy())

    pred_names = get_class_name(weights, pred_id)
    # plot_images(org_images, pred_names)

    saliencies = get_saliency_map(model, x_test)
    visualize_gradients(org_images, pred_names, saliencies)


if __name__ == '__main__':
    main()