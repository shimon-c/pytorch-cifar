'''Train CIFAR10 with PyTorch.'''
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
import onnx
import numpy as np
from utils import progress_bar
#from sc_resnet import ResNet50
#from sc_resnet import ResNet50


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--relu6', type=int, help='relu6 option')
parser.add_argument('--export_onnx', type=str, default='', help='model to export to onnx')
parser.add_argument('--nepochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--sc_resnet', type=int, default=0, help='Use sc resnet version')
args = parser.parse_args()

cpu_device = 'cpu'
def quantize_static(myModel):
    # set quantization config for server (x86)
    myModel.qconfig = torch.quantization.get_default_config('fbgemm')

    # insert observers
    torch.quantization.prepare(myModel, inplace=True)
    # Calibrate the model and collect statistics

    # convert to quantized version
    torch.quantization.convert(myModel, inplace=True)
    return myModel

def quantize(model_fp32):
    model_fp32.to(cpu_device)
    quant_set = {torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d}
    print(f'sizeof fp32: {sys.getsizeof((model_fp32))}')
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,  # the original model
        quant_set,  # a set of layers to dynamically quantize
        dtype=torch.qint8)
    batch_size=32
    print(f'sizeof fp8: {sys.getsizeof(model_int8)}')
    x = torch.randn(batch_size, 3, 32, 32, requires_grad=True, dtype=torch.float32)
    y = model_int8(x)
    return model_int8

#model_fp32 = ResNet18()
#quantize(model_fp32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Function to export to onnx
def export_onnx(cp_name):
    net = ResNet18()
    checkpoint = torch.load(cp_name)
    net_item = checkpoint['net']
    net.load_state_dict(net_item)
    onnx_name = cp_name + '.onnx'
    # Input to the model
    batch_size = 32
    x = torch.randn(batch_size, 3, 32, 32, requires_grad=True, dtype=torch.float32)
    torch_out = net(x)

    # Export the model
    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print(f'onnx model name:{onnx_name}')
    return onnx_name
    #sys.exit(1)

if args.export_onnx != '' and os.path.isfile(args.export_onnx):
    net = ResNet18()
    checkpoint = torch.load(args.export_onnx)
    net_item = checkpoint['net']
    net.load_state_dict(net_item)
    onnx_name = args.export_onnx +'.onnx'
    # Input to the model
    batch_size = 32
    x = torch.randn(batch_size, 3, 32, 32, requires_grad=True, dtype=torch.float32)
    torch_out = net(x)

    # Export the model
    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print(f'onnx model name:{onnx_name}')
    sys.exit(1)

def compute_accuracy(net, to_dev_flg=True):
    global best_acc
    net.eval()
    if to_dev_flg:
        net.to(device)
    test_loss = 0
    correct = 0
    total = 0
    L = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if to_dev_flg:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    clname = str(net.__class__)
    print(f'class:{clname} \t accuracy:{acc}')
    return acc


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

num_workers = 1
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True) #num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False) # num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class Logger:
    def __init__(self, filename):
        self.filename = filename

        self.file = open(filename, "w")
    def __call__(self, test_name=None, epoch=0, correct=0, total=0):
        acc = correct/total
        msg = f'{test_name}, epoch:{epoch}, correct:{correct}/{total}, accuracy:{acc}\n'
        self.file.write(msg)
        self.file.flush()
    def newl(self):
        self.file.write('\n')
    def writeline(self, ln):
        self.file.write(ln)
        self.newl()

# Model
print('==> Building model..')
# net = VGG('VGG19')
resnet.relu6_flg = True if args.relu6 else False
if args.sc_resnet:
    net = sc_resnet.ResNet50()
else:
    net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda' and num_workers>1:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

logger_file = 'C:/Users/shimon.cohen/PycharmProjects/pytorch-cifar/logdir/log_relu.txt'
checkpoint_name = './checkpoint/ckpt_relu.pth'
if args.relu6:
    logger_file = 'C:/Users/shimon.cohen/PycharmProjects/pytorch-cifar/logdir/log_relu6.txt'
    checkpoint_name ='./checkpoint/ckpt_relu6.pth'
if args.sc_resnet:
    logger_file = 'C:/Users/shimon.cohen/PycharmProjects/pytorch-cifar/logdir/sc_log_relu.txt'
    checkpoint_name = './checkpoint/sc_ckpt_relu.pth'
    optimizer = optim.Adam(lr=1e-5, params=net.parameters())
    print('sc_resnet')


lobj = Logger(logger_file)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    L = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f'train epoch:{epoch}, correct:{correct}/{L}, accuracy:{correct/total}')
    lobj(test_name='train',epoch=epoch, correct=correct,total=total)
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    L = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'test epoch:{epoch}, correct:{correct}/{L}, accuracy:{correct/total}')
        lobj(test_name='test', epoch=epoch, correct=correct, total=total)
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, checkpoint_name)
        best_acc = acc

clname = str(net.__class__)
lobj.writeline(clname)
print(f'net type: {clname}')
end_epoch = args.nepochs
for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)
    lobj.newl()
    scheduler.step()

# Load from checkpoint


acc = compute_accuracy(net)
acc_str = f'onnx model accuracy: {acc} ,modelsize: {sys.getsizeof(net)}'
lobj.writeline(acc_str)
net8i = quantize(net)
#net8i = quantize_static(net)
acc = compute_accuracy(net8i, to_dev_flg=False)
acc_str = f'onnx quantized model accuracy: {acc} modelsize: {sys.getsizeof(net8i)}'
lobj.writeline(acc_str)

onnx_name = export_onnx(checkpoint_name)
net = onnx.load(onnx_name)

#outputs = net.run(np.random.randn(10, 3, 32, 32).astype(np.float32))
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.
#print(outputs[0])

