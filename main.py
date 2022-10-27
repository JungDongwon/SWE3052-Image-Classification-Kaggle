import numpy as np
import pandas as pd

from dataset import SSL_Dataset
from model import CNN
from utils import save_prediction, plot_accuracy

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models 
import torch.nn as nn  
from time import time 
from sklearn.model_selection import train_test_split

"""
Please use this code as a guideline. 
Feel free to create your own code for training, testing, ... etc.
But for creating "submission.csv" file, utilizing this code is highly recommended.
"""

class Trainer:
    def __init__(self, model, device, weight_path, model_name, patience, momentum, weight_decay, learning_rate, print_every, max_alpha):

        self.patience = patience
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = learning_rate
        self.print_every = print_every
        self.max_alpha = max_alpha
        self.round = 1
        self.step = 0
    
        self.best_acc = 0
        self.best_epoch = 0
        self.crnt_epoch = 0 
        self.endure = 0 
        self.stop_flag = False
        self.num_class = 10
        self.device = device
        self.weight_path = weight_path
        self.model_name = model_name

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss() 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        self.train_acc = []
        self.valid_acc = []
        
    # test
    def _test(self, mode, data_loader, graph=False):
       
        test_preds = []
        self.model.eval()
        correct = 0
        total = 0
        
        if mode == 'Valid':
            with torch.no_grad():
                for batch_data in data_loader: 
                    batch_x, batch_y = batch_data 
                    inputs, targets = batch_x.to(self.device), batch_y.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1) 

                    total += targets.size(0)
                    correct += predicted.eq(targets).cpu().sum().item()
                    if self.device == 'cuda':
                        test_preds += predicted.detach().cpu().numpy().tolist()
                    else:
                        test_preds += predicted.detach().numpy().tolist()

            total_acc = correct / total

            print("| \033[31m%s Epoch #%d\t Accuracy: %.2f%%\033[0m" %(mode, self.crnt_epoch+1, 100.*total_acc))
            if self.crnt_epoch % self.print_every == 0 and graph == True : 
                self.valid_acc.append(total_acc)
            if self.best_acc < total_acc:
                print('| \033[32mBest Accuracy updated (%.2f => %.2f)\033[0m\n' % (100.*self.best_acc, 100.*total_acc))
                self.best_acc = total_acc
                self.best_epoch = self.crnt_epoch
                self.endure = 0
                # Save best model
                torch.save(self.model.state_dict(), self.weight_path+self.model_name)
            else:
                self.endure += 1
                print(f'| Endure {self.endure} out of {self.patience}\n')
                if self.endure >= self.patience:
                    print('Early stop triggered...!')
                    self.stop_flag = True

        if mode == 'Test':
            print('Predicting Starts...')
            with torch.no_grad():
                for batch_data in data_loader: 
                    batch_x = batch_data 
                    inputs = batch_x.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    if self.device == 'cuda':
                        test_preds += predicted.detach().cpu().numpy().tolist()
                    else:
                        test_preds += predicted.detach().numpy().tolist()

            return test_preds, self.crnt_epoch, self.train_acc, self.valid_acc

    # train
    def label_train(self, labeled_trainloader,labeled_validloader, graph):
        self.model.train()
        print('Labeled Training Starts...')
        total = 0
        correct = 0 
        epochs = 100
        for epoch in range(epochs):
            self.crnt_epoch = epoch 
            for batch_data in labeled_trainloader:
                batch_x, batch_y = batch_data 
                batch_size = batch_x.size(0) 
                batch_y = torch.zeros(batch_size, self.num_class).scatter_(1, batch_y.view(-1,1), 1) 
                inputs_l, targets_l = batch_x.to(self.device), batch_y.long().to(self.device) 
            
                outputs = self.model(inputs_l)
                loss = self.loss_fn(outputs,torch.argmax(targets_l, dim=-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Epoch: {epoch+1}, Loss:{loss:.4f}")
                
                _, predicted = torch.max(outputs, 1)

                total += targets_l.size(0)
                correct += predicted.eq(torch.argmax(targets_l, dim=-1)).cpu().sum().item()

            total_acc = correct / total
            #if self.crnt_epoch % self.print_every == 0 : 
            #    self.train_acc.append(total_acc)
            print("\n| \033[31mTrain Epoch #%d\t Accuracy: %.2f%%\033[0m" %(self.crnt_epoch+1, 100.*total_acc))
            pred = self._test("Valid", labeled_validloader, graph)
            if self.stop_flag : 
                break
        self.stop_flag = False
        self.best_acc=0
        self.endure=0
        print('Labeled Training Finished...!!')
        
    def unlabel_train(self, unlabeled_trainloader, labeled_trainloader, labeled_validloader, graph):
        self.model.train()
        print('Unlabeled Training Starts...')
        total = 0
        correct = 0 
        epochs = 150
        sharpen = 0.5
        for epoch in range(epochs):
            self.crnt_epoch = epoch 
            for batch_idx, batch_data in enumerate(unlabeled_trainloader):
                batch_x = batch_data 
                inputs_l = batch_x.to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(inputs_l)
                    _, pseudo_labels = torch.max(outputs, 1)

                self.model.train()
                outputs = self.model(inputs_l)
                alpha_weight = (self.step/500) * self.max_alpha
                loss = self.loss_fn(outputs, pseudo_labels) * alpha_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Round: {self.round}, Epoch: {epoch+1}, Step: {self.step}, Alpha: {alpha_weight}, Loss:{loss:.4f}")
                
                _, predicted = torch.max(outputs, 1)

                total += pseudo_labels.size(0)
                correct += predicted.eq(pseudo_labels).cpu().sum().item()

                if batch_idx % 50 == 0:
                    for batch_data in labeled_trainloader:
                        batch_x, batch_y = batch_data 
                        batch_size = batch_x.size(0) 
                        batch_y = torch.zeros(batch_size, self.num_class).scatter_(1, batch_y.view(-1,1), 1) 
                        inputs_l, targets_l = batch_x.to(self.device), batch_y.long().to(self.device) 

                        outputs = self.model(inputs_l)
                        loss = self.loss_fn(outputs,torch.argmax(targets_l, dim=-1))

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    self.step += 1
                    
            total_acc = correct / total
            if self.crnt_epoch % self.print_every == 0 and graph == True: 
                self.train_acc.append(total_acc)
            pred = self._test("Valid", labeled_validloader, graph)
            if self.stop_flag : 
                break
        self.stop_flag = False
        self.best_acc=0
        self.endure=0
        self.round += 1
        print('Unlabeled Training Finished...!!')

def load_labeled_data(mode, root):
    data = []
    targets = []
    idx = []
    labeled_train = "kaggle_data/annotation/train_labeled_filelist.txt" 

    if mode == 'labeled_train':
        flist = root + labeled_train
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                imgdata, clean_label = line.strip().split()
                data.append(imgdata)
                targets.append(float(clean_label))
            targets = torch.LongTensor(targets)
            return data, targets

        
def load_unlabeled_data(mode, root):
    data = []
    targets = []
    idx = []
    unlabeled_test = "kaggle_data/annotation/test_filelist.txt"
    unlabeled_train = "kaggle_data/annotation/train_unlabeled_filelist.txt" 

    if mode == 'test': 
        flist = root + unlabeled_test
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                imgdata = line.strip()
                data.append(imgdata)
                idx.append(imgdata.split('/')[2][:-4])
            return idx, data
    elif mode == 'unlabeled_train':
        flist = root + unlabeled_train
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                imgdata = line.strip()
                data.append(imgdata)
                idx.append(imgdata.split('/')[2][:-4])
            return idx, data

def main():

    #################### EDIT HERE ####################
    """
    You can change any values of hyper-parameter below.
    *test_only: If this parameter is True, you can just test with a model that already exists without training step. 
    (for your time saving..!) 
    """
    random_seed=1

    patience = 5
    momentum = 0.9
    #weight_decay = 5e-4
    weight_decay = 5e-3
    learning_rate = 0.001
    #learning_rate = 0.0005

    print_every = 1
    train_batch = 256
    test_batch = 256
    valid_ratio = 0.1

    model_name = 'my_model.p'

    test_only = False
    max_alpha = 2
    ###################################################

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        device = 'cuda' 
        torch.cuda.manual_seed_all(random_seed) 
        torch.backends.cudnn.deterministic = True
    else :
        device = 'cpu'
    
    weight_path = './best_model/'

    train_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.RandomHorizontalFlip(),

        transforms.RandomCrop(128),

        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data, labels = load_labeled_data('labeled_train', root='../')
    train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, test_size=valid_ratio, random_state=random_seed)

    train_labeled_dataset = SSL_Dataset(train_data, train_labels, transform=train_transform, mode='labeled_train')
    valid_labeled_dataset = SSL_Dataset(valid_data, valid_labels, transform=test_transform, mode='labeled_train')
    
    test_idx, test_data = load_unlabeled_data(mode="test", root='../')
    test_labeled_dataset = SSL_Dataset(test_data, None,transform=test_transform, mode="test") 
    
    unlabled_train_idx, unlabled_train_data = load_unlabeled_data(mode="unlabeled_train", root='../')
    train_unlabeled_dataset = SSL_Dataset(unlabled_train_data, None,transform=test_transform, mode="unlabeled_train") 
    
    labeled_trainloader = DataLoader(train_labeled_dataset, batch_size=train_batch, shuffle=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_dataset, batch_size=train_batch, shuffle=True)
    labeled_validloader = DataLoader(valid_labeled_dataset, batch_size=train_batch, shuffle=False)
    labeled_testloader = DataLoader(test_labeled_dataset, batch_size=test_batch, shuffle=False)

    #################### EDIT HERE ####################

    #model = models.densenet121()
    #model.classifier = nn.Linear(1024, 10).to(device)
    
    model = models.resnet50(pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, 10).to(device)

    #model = CNN().to(device)

    ###################################################
    
    trainer = Trainer(model, device, weight_path, model_name, patience, momentum, weight_decay, learning_rate, print_every, max_alpha)

    if test_only == False:
        print(f"# Train data: {len(train_labeled_dataset)}, # Valid data: {len(valid_labeled_dataset)}")
        train_start = time()
        
        trainer.label_train(labeled_trainloader,labeled_validloader, graph=False)
        trainer.unlabel_train(unlabeled_trainloader, labeled_trainloader, labeled_validloader, graph=True)
        
        train_elapsed = time() - train_start
        print('Train Time: %.4f\n' % train_elapsed)

    print(f"# Test data: {len(test_labeled_dataset)}")
    model.load_state_dict(torch.load(weight_path+model_name))
    pred, num_epochs, train_acc, valid_acc = trainer._test("Test", labeled_testloader)
    save_prediction(weight_path, pred, test_idx)
    plot_accuracy(print_every, weight_path, num_epochs, train_acc, valid_acc)

if __name__ == "__main__":
    main()