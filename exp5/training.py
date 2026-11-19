import torch
from torch.nn import functional as F
from torchvision import datasets, transforms


def make_loaders(data_dir, batch_size):
    transform = transforms.ToTensor()
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    return tuple(torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
                 for dataset in (train, test))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += output.argmax(1).eq(y).sum().item()
        count += len(y)
    return total_loss / count, correct / count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    batch_loss = weighted_loss = correct = count = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output, y).item()
        batch_loss += loss
        weighted_loss += loss * len(y)
        pred = output.argmax(1)
        correct += pred.eq(y).sum().item()
        count += len(y)
        confusion += torch.bincount((y * 10 + pred).cpu(), minlength=100).reshape(10, 10)
        last_images, last_labels = x[:5].detach().cpu(), y[:5].detach().cpu()
    return ({"test_loss_pdf": batch_loss / len(loader),
             "test_loss_sample_mean": weighted_loss / count,
             "test_accuracy": correct / count, "correct": correct},
            confusion.numpy(), last_images, last_labels)


@torch.no_grad()
def predict_samples(model, dataset, indices, device):
    model.eval()
    images = torch.stack([dataset[int(i)][0] for i in indices])
    labels = [int(dataset.targets[int(i)]) for i in indices]
    predictions = model(images.to(device)).argmax(1).cpu().tolist()
    return images.squeeze(1).numpy(), labels, predictions


@torch.no_grad()
def extract_features(model, images, device):
    model.eval()
    feature1, feature2 = model.feature_maps(images.to(device))
    return feature1.cpu().numpy(), feature2.cpu().numpy()
