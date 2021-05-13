import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import tqdm
import os

# select device utils
def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

# Net class
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        in_size = x.size(0)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, (2,2), stride=2)        # change sigmoid(in original LeNet5) to relu
        out = F.max_pool2d(F.relu(self.conv2(out)), (2,2))
        out = out.view(in_size, -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out



# train function
def train():
    epochs = args.epochs
    batch_size = args.batch_size

    # Trainloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batches = len(train_loader)

    # Init model, optimizer, loss function
    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()   # to use BN and dropout
        mloss = 0   # mean loss
        print(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'loss', 'targets'))
        pbar = tqdm.tqdm(enumerate(train_loader), total=batches)  # progress bar
        for i, (imgs, targets) in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            # Print
            mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 2) % ('%g/%g' % (epoch, epochs - 1), mem, mloss, len(targets))
            pbar.set_description(s)
    print('%g epochs training completed.\n' % epochs)
    return model


# test function
def test(model):
    batch_size = args.batch_size
    # Testloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST("./mnist", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    s = ('%30s' + '%10s') % ('Images', 'Acc')
    correct = 0
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(tqdm.tqdm(test_loader, desc=s)):
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            pred = output.max(1, keepdim=True)[1] # find 
            correct += pred.eq(targets.view_as(pred)).sum().item()
    # Print results
    acc = correct / len(test_loader.dataset)
    print(('%30s' + '%10.3g') % (len(test_loader.dataset), acc))
    print('test completed.')
            
 


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--epochs', type=int, default=10)
    paser.add_argument('--batch-size', type=int, default=64)
    paser.add_argument('--device', default='')
    args = paser.parse_args()
    print(args)
    device = select_device(args.device)
    model = train()
    test(model)







