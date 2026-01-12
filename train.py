from model import CNN_LSTM
from config_parser import parse_file
from dataset import preprocess_data
import os
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from customdata import CustomDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

config_file = parse_file('config.ini')
cnn_lstm_model = CNN_LSTM(config_file)
print(cnn_lstm_model)

if config_file['use_existing_preprocessesd_data']:
    windows_arr = np.load(config_file['preprocessed_windows_data'])
    labels_array = np.load(config_file['preprocessed_labels_data'])
    with open(config_file['class_idx_file'],'rb') as obj:
        class_idx = pickle.load(obj)

else:
    class_idx, windows_arr,labels_array = preprocess_data(config_file)
    with open(config_file['class_idx_file'],'wb') as obj:
        pickle.dump(class_idx ,obj)

indices = np.random.permutation(len(windows_arr))
cnn_inp_shuffled = windows_arr[indices]
cnn_labels_shuffled = labels_array[indices]

print('Classes present - ',Counter(labels_array))


cnn_inp_shuffled = torch.tensor(cnn_inp_shuffled, dtype=torch.float32)  # Inputs as Float
cnn_labels_shuffled = torch.tensor(cnn_labels_shuffled, dtype=torch.long) 

X_train, X_test, y_train, y_test = train_test_split(

cnn_inp_shuffled, cnn_labels_shuffled, test_size=config_file['train_test_split'], random_state=42)

dataset = CustomDataset(X_train, y_train)

# Create a DataLoader for batch processing
batch_size = config_file['batch_size']  # You can set this to your desired batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'])  # Adam optimizer

num_epochs = config_file['num_epochs']  # Set the number of epochs

num_training_steps = len(data_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=num_training_steps, 
                        pct_start=0.3, anneal_strategy='linear')

alpha = 1
gamma = 2
lr_li = []
accuracy_li = []
avg_loss_li = []
for epoch in range(num_epochs):
    cnn_lstm_model.train()  # Set the model to training mode
    total_loss = 0
    total_correct = 0

    for batch_inputs, batch_labels in data_loader:
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        outputs = cnn_lstm_model(batch_inputs)  # Shape: (batch_size, num_classes)
        
        # Calculate loss
        loss = criterion(outputs, batch_labels)
        #total_loss += loss.item()

        # Backward pass
        #Focal loss
        pt = torch.exp(-loss)
        F_loss = alpha * (1-pt)**gamma * loss
        total_loss += F_loss
        F_loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        scheduler.step()

        # Calculate accuracy
        predicted = torch.argmax(nn.Softmax(dim=1)(outputs),1)  # Get the predicted class
        # print(Counter(list(predicted)),'\npredicted - ',(predicted == batch_labels).sum().item()) 
        total_correct += (predicted == batch_labels).sum().item()
        
        
    current_lr = optimizer.param_groups[0]['lr']
    lr_li.append(current_lr)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}')
    avg_loss = total_loss / len(data_loader)
    avg_loss_li.append(avg_loss)
    # print(total_correct)
    accuracy = total_correct / len(dataset) * 100
    accuracy_li.append(accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

torch.save(cnn_lstm_model.state_dict(), config_file['model_path'])

plt.figure(0)
plt.plot([float(v) for v in avg_loss_li])
plt.title('Average loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(config_file['loss_graph'])

plt.figure(1)
plt.plot(lr_li)
plt.title('learning rate')
plt.xlabel('Epoch')
plt.ylabel('Lr')
plt.savefig(config_file['lr_graph'])

plt.figure(2)
plt.plot(accuracy_li)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.savefig(config_file['acc_graph'])

pred = cnn_lstm_model(X_test)
predicted = torch.argmax(nn.Softmax(dim=1)(pred),1)

pre,rec,f1,support = precision_recall_fscore_support(y_test, predicted)

intersection_matrix = np.vstack([pre, rec,f1,support])
print('Precision : ',pre)
print('Recall : ',rec)
print('F1score : ',f1)

plt.figure(4)
plt.figure(figsize=(10,10))
plt.matshow(intersection_matrix[:config_file['num_classes'],:],cmap=plt.cm.Blues)

for i in range(config_file['num_classes']):
    for j in range(config_file['num_classes']):
        c = intersection_matrix[j,i]
        plt.text(i, j, str(round(c,4)), va='center', ha='center')

plt.yticks(np.arange(config_file['num_classes']),['Precision','Recall','F1score'])
plt.xticks(np.arange(config_file['num_classes']),[i[0]+'\n'+str(i[1]) for i in zip(list(class_idx.keys()) , intersection_matrix[config_file['num_classes'],:])])
plt.savefig('Precision_recall_f1score.png')
# plt.show()

cnf_mat = confusion_matrix(y_test,predicted)

print('Confusion matrix:\n',cnf_mat)

plt.figure(figsize=(10,10))
plt.matshow(cnf_mat,cmap=plt.cm.Blues)

for i in range(config_file['num_classes']):
    for j in range(config_file['num_classes']):
        c = cnf_mat[j,i]
        plt.text(i, j, str(round(c,4)), va='center', ha='center')

plt.yticks(np.arange(config_file['num_classes']),list(class_idx.keys()))
plt.xticks(np.arange(config_file['num_classes']),[i[0]+'\n'+str(i[1]) for i in zip(list(class_idx.keys()) , cnf_mat)])
plt.xlabel('Predicted')
plt.ylabel('Actual')
# plt.tight_layout()
plt.savefig('Confusion matrix.png')


print(classification_report(y_test, predicted, target_names=class_idx.keys()))

        
        