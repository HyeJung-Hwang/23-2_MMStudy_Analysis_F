from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model,inputs,targets,epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(inputs)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, targets)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    
        loss.backward()
        optimizer.step()

