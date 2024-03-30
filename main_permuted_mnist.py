import torch
import torchvision.transforms as transforms
from data.permuted_mnist import Permuted_MNIST
from utils import device
from model_permuted_mnist import MLP, MLP_VI, ConvNet, ConvNet_VI

EXPERIMENT = "permuted_mnist"
NB_EPOCHS = 100
NB_SAMPLES = 100
CORESET_SIZE = 200
NB_HEADS = 1
#HIDDEN_DIM = 100  # For the MLP
HIDDEN_DIM = 4     # For the CNN
OUT_DIM = 10
NB_TASKS = 10

trainset = Permuted_MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor(), coreset_size=CORESET_SIZE)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = Permuted_MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor(), coreset_size=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

print(device)

trainset.select_task(0)
testset.select_task(0)
#model_init = MLP(in_dim=28*28, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
model_init = ConvNet(in_channels=1, hidden_channels=HIDDEN_DIM, out_channels=OUT_DIM)
model_init.fit(trainloader=trainloader, nb_epochs=NB_EPOCHS)
correct, total, acc = model_init.test(testloader)
print('First Model: Acc = %.3f%% (%d/%d)\n' % (acc, correct, total))

#model = MLP_VI(in_dim=28*28, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, model_init=model_init, mc_samples=NB_SAMPLES)
model = ConvNet_VI(in_channels=1, hidden_channels=HIDDEN_DIM, out_channels=OUT_DIM, model_init=model_init, mc_samples=NB_SAMPLES)
for i in range(NB_TASKS):
    task_size = trainset.select_task(i, update_coreset=True)
    model.set_task_size(task_size)
    model.update_prior(initial=True if i==0 else False)
    model.fit(trainloader=trainloader, nb_epochs=NB_EPOCHS)
    with torch.no_grad():
        print('---------------------------- TASK %d BEFORE CORESET ----------------------------' % (i+1))
        for j in range(i+1):
            testset.select_task(j)
            correct, total, acc = model.test(testloader=testloader)
            print('Task %d: Acc = %.3f%% (%d/%d)' % (j + 1, acc, correct, total))
        print('-------------------------------------------------------------------------------')
    model.update_prior()

    if trainset.coreset_size > 0:
        trainset.train_on_coreset()
        task_size = trainset.select_task(task_id=-1)  # data is merged for coreset training and task_id is irrelevant
        #model_finetune = MLP_VI(in_dim=28*28, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, model_init=model, mc_samples=NB_SAMPLES, identical=True)
        model_finetune = ConvNet_VI(in_channels=1, hidden_channels=HIDDEN_DIM, out_channels=OUT_DIM, model_init=model, mc_samples=NB_SAMPLES, identical=True)
        model_finetune.set_task_size(task_size)
        model_finetune.fit(trainloader=trainloader, nb_epochs=NB_EPOCHS)

        print('----------------------------- TASK %d AFTER CORESET -----------------------------' % (i+1))

        for j in range(i + 1):    
            with torch.no_grad():
                testset.select_task(j)
                correct, total, acc = model_finetune.test(testloader=testloader)
                print('Task %d: Acc = %.3f%% (%d/%d)' % (j + 1, acc, correct, total))
        print('------------------------------------- END -------------------------------------\n')
        trainset.train_on_full_dataset()