import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10 
from torchvision import transforms
from models.resnext_checkpoint import ResNeXt29_4x64d
from tqdm import trange
import argparse
from torch.cuda.amp import GradScaler



def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.*correct / total


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight)



if __name__ ==  "__main__":

    parser = argparse.ArgumentParser(description='Training Parameter')

    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")

    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()

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

    train_data = CIFAR10(root = "./data/", transform = transform_train, train = True, download = True)
    train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True, num_workers=args.num_worker)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    model = ResNeXt29_4x64d()

    model.apply(init_weights)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    training_loss = []
    training_acc = []
    testing_acc = []

    if args.half_precision:
        scaler = GradScaler()

    with torch.autograd.profiler.emit_nvtx():
        torch.cuda.nvtx.range_push("training resnext with checkpoint")
        for i in trange(args.num_epoch): 
            torch.cuda.nvtx.range_push(f"epoch {i}")
            model.train()
            running_loss = 0
            correct = 0

            for j, (x, y) in enumerate(train_data_loader):
                if j > 0:
                    torch.cuda.nvtx.range_pop() #data load

                torch.cuda.nvtx.range_push(f"epoch {i} - step {j}")

                torch.cuda.nvtx.range_push("data copy")
                x = x.cuda()
                y = y.cuda()
                torch.cuda.nvtx.range_pop() #data copy

                torch.cuda.nvtx.range_push("zero grad")
                optimizer.zero_grad()
                torch.cuda.nvtx.range_pop() #zero grad

                if args.half_precision:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        torch.cuda.nvtx.range_push("forward")
                        y_hat = model(x)
                        torch.cuda.nvtx.range_pop() #forward

                        l = loss(y_hat, y)
                    
                    scaler.scale(l).backward()

                    torch.cuda.nvtx.range_push("optimizer")
                    scaler.step(optimizer)
                    torch.cuda.nvtx.range_pop() #optimizer

                    scaler.update()

                else: 
                    torch.cuda.nvtx.range_push("forward")
                    y_hat = model(x)
                    torch.cuda.nvtx.range_pop() #forward

                    l = loss(y_hat, y)
                    l.backward()

                    torch.cuda.nvtx.range_push("optimizer")
                    optimizer.step()
                    torch.cuda.nvtx.range_pop() #optimizer

                running_loss += l.item() * x.size(0)
                _, predicted = y_hat.max(1)
                correct += predicted.eq(y).sum().item()

                torch.cuda.nvtx.range_pop() #step

                if j < len(train_data_loader)-1:
                    torch.cuda.nvtx.range_push("data load")

            training_loss.append(running_loss / 50000)
            training_acc.append(100*correct / 50000)

            torch.cuda.nvtx.range_push("validation")
            testing_acc.append(test(model))
            torch.cuda.nvtx.range_pop() #validation

            torch.cuda.nvtx.range_pop() # epoch

        torch.cuda.nvtx.range_pop() # train


    if args.half_precision:
        torch.save(model.state_dict(), f"model_history/resnext_{args.num_worker}_worker_{args.batch_size}_fp16_checkpoint.pt")
        np.savez(f"model_history/resnext_{args.num_worker}_worker_{args.batch_size}_fp16_checkpoint.npz", loss=training_loss, acc=training_acc, test_acc=testing_acc)
    else: 
        torch.save(model.state_dict(), f"model_history/resnext_{args.num_worker}_worker_{args.batch_size}_checkpoint.pt")
        np.savez(f"model_history/resnext_{args.num_worker}_worker_{args.batch_size}_checkpoint.npz", loss=training_loss, acc=training_acc, test_acc=testing_acc)







