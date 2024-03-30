import torch
import torchvision.transforms as transforms
from data.split_mnist import Split_MNIST
from utils import device
from model_split_mnist import MLP, MLP_VI, ConvNet, ConvNet_VI

EXPERIMENT = "split_mnist"
NB_EPOCHS = 120
NB_SAMPLES = 100
CORESET_SIZE = 40
NB_HEADS = 5
#HIDDEN_DIM = 256  # For the MLP
HIDDEN_DIM = 9     # For the CNN
OUT_DIM = 2
NB_TASKS = 5
BATCH_SIZE = 1024

# The training set is initialized with the task 0-1.
trainset = Split_MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor(), coreset_size=CORESET_SIZE)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = Split_MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor(), coreset_size=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print(device)

trainset.select_task(0)
testset.select_task(0)
#model_init = MLP(in_dim=28*28, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
model_init = ConvNet(in_channels=1, hidden_channels=HIDDEN_DIM, out_channels=OUT_DIM)
model_init.fit(trainloader=trainloader, nb_epochs=NB_EPOCHS)
correct, total, acc = model_init.test(testloader, 0)
print('First Model: Acc = %.3f%% (%d/%d)\n' % (acc, correct, total))

#model = MLP_VI(in_dim=28*28, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, nb_heads=NB_HEADS, model_init=model_init, mc_samples=NB_SAMPLES)
model = ConvNet_VI(in_channels=1, hidden_channels=HIDDEN_DIM, out_channels=OUT_DIM, nb_heads=NB_HEADS, model_init=model_init, mc_samples=NB_SAMPLES)
for i in range(NB_TASKS):
    task_size = trainset.select_task(i, update_coreset=True)
    model.set_task_size(task_size)
    model.update_prior(i, new_task=True, initial=True if i==0 else False)
    model.fit(trainloader=trainloader, nb_epochs=NB_EPOCHS, task_id=i)
    with torch.no_grad():
        print('----------------------- TASK BEFORE CORESET %d : (%d, %d) -----------------------' % (i+1, 2 * i, 2 * i + 1))
        for j in range(i+1):
            testset.select_task(j)
            correct, total, acc = model.test(testloader=testloader, task_id=j)
            print('Task (%d, %d): Acc = %.3f%% (%d/%d)' % (2 * j, 2 * j + 1, acc, correct, total))
        print('--------------------------------------------------------------------------------')
    model.update_prior(i)

    if trainset.coreset_size > 0:
        trainset.train_on_coreset()
        print('------------------------- TASK AFTER CORESET %d : (%d, %d) ------------------------' % (i+1, 2 * i, 2 * i + 1))
        for j in range(i + 1):
            task_size = trainset.select_task(j)
            #model_finetune = MLP_VI(in_dim=28*28, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, nb_heads=NB_HEADS, model_init=model, mc_samples=NB_SAMPLES, identical=True)
            model_finetune = ConvNet_VI(in_channels=1, hidden_channels=HIDDEN_DIM, out_channels=OUT_DIM, nb_heads=NB_HEADS, model_init=model, mc_samples=NB_SAMPLES, identical=True)
            model_finetune.set_task_size(task_size)
            model_finetune.fit(trainloader=trainloader, nb_epochs=NB_EPOCHS, task_id=j)

            with torch.no_grad():
                testset.select_task(j)
                correct, total, acc = model_finetune.test(testloader=testloader, task_id=j)
                print('Task (%d, %d): Acc = %.3f%% (%d/%d)' % (2 * j, 2 * j + 1, acc, correct, total))
        print('-------------------------------------- END -------------------------------------')
        print()
        trainset.train_on_full_dataset()