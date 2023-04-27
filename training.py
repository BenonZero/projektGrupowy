# TODO https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import sys
import datetime
# from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, optim, save
from torchaudio.datasets import SPEECHCOMMANDS
import os
# from os import walk

from E2E_KS_Attention_Energy_Scorer.source.QED import QueryEncDec

# if __name__ != "__main__":
#     exit()

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    # "C:\\Users\\kubak\\INF_SEM5\\PG\\repo\\projektGrupowy\\speech_commands_v0.02"
    # TODO : jak podzielic dane na uczace testujace i epoki

    training_loader = []
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        inputs = "bird"
        target = inputs.lower()
        target = [float(ord(i)) for i in target]
        target = [target[i % len(target)] for i in range(256)]
        target = tensor([target])
        
        outputs = model(inputs, training = True)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



# roznice w datasetach
# roznice w modelach

args = sys.argv[1:]
if len(args) != 1:
    exit()

model = None
if args[0] == "1":
    # obsluga metody 1.
    model = "obsluga metody 1"
elif args[0] == "2":
    # obsluga metody 2.
    model = QueryEncDec()
elif args[0] == "3":
    # obsluga metody 3.
    model = "obsluga metody 3"
else:
    print("[indian accent] sorry sir but your input is INCORRECT")
    exit()
    
print("training: " + str(model))



# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")
exit()

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]



loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'../runs/QED_trainer_{timestamp}')
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    # TODO ogarnac czy wszystko dziala z train(False) (metadane)
    model.train(False)

    running_vloss = 0.0
    # TODO validation_loader = torch.utils.data.DataLoader
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = '../model_{}_{}'.format(timestamp, epoch_number)
        save(model.state_dict(), model_path)
    
    epoch_number += 1