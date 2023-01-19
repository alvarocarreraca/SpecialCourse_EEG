import csv
import numpy as np
from scipy import signal
from mat4py import loadmat
from cnn import Net
from cnnExtended import NetExtended
from cnn1D import NetExtended1D
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import torch.optim as optCNN
import torch.nn as nn
from dataset import Dataset
import torch
import timeit
import random
# Bool
is1D = True
useAllTrial = False


def loadControlPatients(fileName):
    X_total = []
    y_total = []
    for patient in fileName:
        data = loadmat('data/Data Control Patients/' + patient)
        duration = 5
        data = data['data']
        datasetX = []
        y = []
        for dt in data:
            aux = np.array(dt['X'])
            fs = dt['fs']
            trial = np.array(dt['trial'])
            y_dt = np.array(dt['y'])
            EEG_signals = aux.transpose()
            sig_Filtered = []
            for sig in EEG_signals:
                sig_bp = bandPass(sig, fs)
                sig_bp_notch = notchFilter(sig_bp, fs)
                sig_Filtered.append(sig_bp_notch)
            X_split = splitTrials(sig_Filtered, trial, fs, duration)
            datasetX.append(X_split)
            y.append(y_dt)
        datasetX = reshapeDataset(datasetX)
        X_wo_CAR = commonAverageReference(datasetX)
        X = []
        for i in range(len(X_wo_CAR)):
            X.append(normalization(X_wo_CAR[i]))
        y = np.concatenate(y, axis=None)
        X_total = X_total + X
        y_total.append(y)
    y_total = np.concatenate(y_total, axis=None)
    return X_total, y_total


def reshapeDataset(dataset):
    # Reshape the dataset (from 20 x 5 matrices to 100 matrices):
    datasetX_reshaped = []
    for ii in dataset:
        for jj in ii:
            datasetX_reshaped.append(jj)
    return datasetX_reshaped


def bandPass(EEGSignal, fs):
    band = np.array([5, 39])
    stopBand = np.array([3, 41])

    N, Wn = signal.cheb2ord(wp=band, ws=stopBand, gpass=3, gstop=60, fs=fs)
    sos = signal.cheby2(N=N, rs=60, Wn=Wn, btype='bandpass', fs=fs, output="sos")
    filtered = signal.sosfilt(sos, EEGSignal)
    return filtered


def notchFilter(EEGSignal, fs):
    band = np.array([48, 52])
    stopBand = np.array([49, 51])

    N, Wn = signal.cheb2ord(wp=band, ws=stopBand, gpass=3, gstop=60, fs=fs)
    sos = signal.cheby2(N=N, rs=60, Wn=Wn, btype='stop', fs=fs, output="sos")
    filtered = signal.sosfilt(sos, EEGSignal)
    return filtered


def takeTrialsUsed(trial, y, y_used):
    # metodo para coger solo las clases que interesan en los pacientes reales (solo para Data Real Patients)
    return 0


def splitTrials(EEG_signals, trials, fs, duration):
    numPoints = fs * duration
    X = []
    for trial_ind in trials:
        trial = []
        for sig in EEG_signals:
            trial.append(sig[trial_ind - 1:trial_ind + numPoints - 1])
        X.append(trial)
    return X


def normalization(EEGSignal):
    EEG_mean = np.std(EEGSignal, axis=1)
    EEG_std = np.mean(EEGSignal, axis=1)
    return (EEGSignal - np.array(EEG_mean, ndmin=2).T) / np.array(EEG_std, ndmin=2).T


def commonAverageReference(ds):
    dataset_wo_CAR = []
    for EEGMatrix in ds:
        CAR = np.mean(EEGMatrix, axis=0)
        matrix_wo_CAR = np.subtract(EEGMatrix, CAR)
        dataset_wo_CAR.append(matrix_wo_CAR)
    return dataset_wo_CAR


def look_for_best_model(results):
    pass


def test_best_model(best_model_index):
    pass


def main():
    start = timeit.timeit()
    num_trials_train = 1000
    num_trials_test = 500
    # control_patients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    control_patients = [1, 2]
    control_patients_train = []
    control_patients_test = []
    for ii in control_patients:
        if ii < 10:
            control_patients_train.append('S0' + str(ii) + 'T.mat')
            control_patients_test.append('S0' + str(ii) + 'E.mat')
        else:
            control_patients_train.append('S' + str(ii) + 'T.mat')
            control_patients_test.append('S' + str(ii) + 'E.mat')
    X_train, y_train = loadControlPatients(control_patients_train)
    X_test, y_test = loadControlPatients(control_patients_test)
    # Pick random trials for train and test:
    if not useAllTrial:
        X_train, y_train = generate_random_dataset(X_train, y_train, num_trials_train)
        X_test, y_test = generate_random_dataset(X_test, y_test, num_trials_test)
    results = train_cnn_model(X_train, y_train, X_test, y_test)
    w = csv.writer(open("output.csv", "w"))
    for key, val in dict.items(results):
        w.writerow([key, val])
    # best_model_index = look_for_best_model(results)
    # test_best_model(best_model_index)
    end = timeit.timeit()
    print('Time of the process:')
    print(end - start)
    # plot_results(results)
    print('Process finished.')


def generate_random_dataset(X_in, y_in, num_trials):
    pair_data = list(zip(X_in, y_in))
    pair_data = random.sample(pair_data, num_trials)
    X_out, y_out = zip(*pair_data)
    return list(X_out), np.asarray(y_out)


def train_cnn_model(X_train_global, y_train_global, X_test_global, y_test_global):
    batch_size = 24
    num_epoch = 500

    # Loss Function:
    criterion = nn.CrossEntropyLoss()

    # Storage of results:
    results = {}

    dataset = Dataset(X_train_global, y_train_global)

    K = 5
    CV = KFold(K, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(CV.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))

        # Load data:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
        params = {'batch_size': batch_size,
                  'shuffle': True}

        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_index)
        training_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        # Generators
        # training_set = Dataset(X_train, y_train)
        # training_generator = torch.utils.data.DataLoader(training_set, **params)

        # test_set = Dataset(X_test, y_test)
        # test_generator = torch.utils.data.DataLoader(test_set, **params)

        # Model:
        num_classes = 2
        x_axis = 3
        y_axis = 640

        # Create NN:
        # model = Net(num_classes, x_axis, y_axis)
        # model = NetExtended(num_classes, x_axis, y_axis)
        model = NetExtended1D(num_classes, x_axis, y_axis)
        model.apply(reset_weights)

        # optimizer = optCNN.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = optCNN.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.95])

        history = {'train_error_per_batch': [], 'correct_per_epoch': [], 'test_error_per_batch': [], 'output_predicted': [], 'output_expected': []}

        # Train:
        model.train()
        train_error_per_batch = []
        correct_per_epoch = []
        for epoch in range(num_epoch):

            running_loss = 0.0
            correct = 0
            for i, data in enumerate(training_generator, 0):
                local_batch, local_labels = data
                # get the inputs
                inputs, labels = local_batch.type(torch.FloatTensor), local_labels.type(torch.LongTensor)
                if not is1D:
                    inputs = inputs[:, None, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(inputs)
                labels = torch.subtract(labels, 1)
                batch_loss = criterion(output, labels)

                batch_loss.backward()
                optimizer.step()

                # total correct
                train_error_per_batch.append(batch_loss.data.detach().numpy())
                preds = np.argmax(output.data.numpy(), axis=-1)
                correct += np.sum(labels.data.numpy() == preds)
                # print statistics
                running_loss += batch_loss.data.numpy()
                if i % 10 == 9:  # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
            correct_per_epoch.append(correct)

        print(np.mean(train_error_per_batch), np.sum(correct_per_epoch) / (num_epoch * len(train_sampler)))
        print('Finished Training: Fold {}'.format(fold + 1))

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)

        # Test:
        model.eval()
        test_error_per_batch = []
        correct_test = 0
        output_predicted = []
        output_expected = []
        for inputs, labels in test_generator:
            inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.LongTensor)
            if not is1D:
                inputs = inputs[:, None, :, :]
            output = model(inputs)  # forward + backward + optimize
            labels = torch.subtract(labels, 1)
            batch_loss = criterion(output, labels)
            test_error_per_batch.append(batch_loss.data.detach().numpy())  # test error
            preds = np.argmax(output.data.detach().numpy(), axis=-1)
            # Update variables:
            correct_test += np.sum(labels.data.numpy() == preds)
            output_predicted.append(preds)
            output_expected.append(labels.data.detach().numpy())

        print('Test error, fold {}:'.format(fold + 1))
        print(np.mean(test_error_per_batch))
        print('Accuracy of the test, fold {}:'.format(fold + 1))
        print(correct_test / len(test_sampler))
        print('Finished Testing: Fold {}'.format(fold + 1))
        history['train_error_per_batch'].append(train_error_per_batch)
        history['correct_per_epoch'].append(correct_per_epoch)
        history['test_error_per_batch'].append(test_error_per_batch)
        history['output_predicted'].append(output_predicted)
        history['output_expected'].append(output_expected)

        results['fold{}'.format(fold + 1)] = history

    return results


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def plot_results(train_error_per_batch, correct_per_epoch, test_error_per_batch, correct_test, output_predicted,
                 output_expected):
    # Plot error per batch:
    x = np.arange(1, len(train_error_per_batch) + 1)
    plt.figure(1)
    plt.plot(x, train_error_per_batch)
    plt.xlabel('Num batches')
    plt.ylabel('Error')
    plt.title('Train error per batch')

    # Plot correct predictions per epoch:
    x = np.arange(1, len(correct_per_epoch) + 1)
    plt.figure(2)
    plt.plot(x, correct_per_epoch)
    plt.xlabel('Num epochs')
    plt.ylabel('Correct')
    plt.title('Correct predictions per epoch (max of 100)')

    # Plot test error per batch:
    x = np.arange(1, len(test_error_per_batch) + 1)
    plt.figure(3)
    plt.plot(x, test_error_per_batch)
    plt.xlabel('Num batches')
    plt.ylabel('Error')
    plt.title('Test error per batch')

    # Plot test predicted vs. real:
    output_predicted_array = []
    for sublist in output_predicted:
        for item in sublist:
            output_predicted_array.append(item)
    output_expected_array = []
    for sublist in output_expected:
        for item in sublist:
            output_expected_array.append(item)

    x = np.arange(1, len(output_expected_array) + 1)
    plt.figure(4)
    plt.plot(x, output_predicted_array, 'o')
    plt.plot(x, output_expected_array, 'o')
    plt.xlabel('Num samples')
    plt.ylabel('Classification')
    plt.title('Test classification predicted vs. expected')
    plt.legend(['Predicted', 'Expected'])

    plt.show()


main()
