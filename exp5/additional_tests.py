import csv
from datetime import datetime
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from lenet_model import Net
from plotting import save_features, save_samples
from training import extract_features, predict_samples


def run(base_dir):
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = base_dir / 'results'
    checkpoint_path = output / 'model_final.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Net().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    dataset = datasets.MNIST(str(base_dir / 'data'), train=False, download=False,
                             transform=transforms.ToTensor())
    groups = []
    logs = [f"Model: epoch {checkpoint['parameters']['epochs'] - 1} | Device: {device}"]
    for number, start in enumerate([20, 40], 1):
        indices = list(range(start, start + 20))
        images, labels, predictions = predict_samples(model, dataset, indices, device)
        correct = sum(label == prediction for label, prediction in zip(labels, predictions))
        prediction_file = f'0{5 + number}_group{number}_predictions.png'
        features_file = f'0{7 + number}_group{number}_features.png'
        save_samples(images, labels, output / prediction_file, predictions)
        feature_images = torch.from_numpy(images[:5]).unsqueeze(1)
        feature1, feature2 = extract_features(model, feature_images, device)
        save_features(feature_images.numpy(), feature1, feature2, labels[:5], output / features_file)
        rows = [{'index': i, 'label': y, 'prediction': p, 'correct': y == p}
                for i, y, p in zip(indices, labels, predictions)]
        with (output / f'group{number}_predictions.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['index', 'label', 'prediction', 'correct'])
            writer.writeheader()
            writer.writerows(rows)
        groups.append({'group': number, 'indices': indices, 'correct': correct,
                       'count': len(indices), 'accuracy': correct / len(indices),
                       'predictions_image': prediction_file, 'features_image': features_file,
                       'samples': rows})
        logs.append(f'Group {number} | test indices {start}-{start + 19} | Correct: {correct}/20 | Accuracy: {correct / 20:.2%}')
        logs.append(f'Labels:      {labels}')
        logs.append(f'Predictions: {predictions}')
    info = {'run_at': datetime.now().astimezone().isoformat(), 'checkpoint': checkpoint_path.name,
            'model_epoch': checkpoint['parameters']['epochs'] - 1,
            'data_split': 'MNIST test', 'groups': groups}
    (output / 'additional_tests.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    text = '\n'.join(logs) + '\n'
    (output / 'additional_tests.txt').write_text(text)
    print(text, end='')
    return info


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
