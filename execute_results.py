import numpy as np
from scipy import signal
from mat4py import loadmat
from cnn import Net
from cnnExtended import NetExtended
from cnn1D import NetExtended1D, NetExtended1D_3sec
from ShallowConvNet import ShallowNet
from cnnTemp_Spac import DeepConvNet
from cnn2D import Net2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.optim as optCNN
import torch.nn as nn
from dataset import Dataset
import torch
import timeit
import random

is1D = True
useAllTrial = False
isDatabaseReady = True


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
    # timeShift = 256
    timeShift = 0
    numPoints = fs * duration
    # numPoints = 522
    X = []
    for trial_ind in trials:
        trial = []
        for sig in EEG_signals:
            trial.append(sig[trial_ind - 1 - timeShift:trial_ind + numPoints - 1 - timeShift])
        X.append(trial)
    return X


def normalization(EEGSignal):
    EEG_mean = np.mean(EEGSignal, axis=1)
    EEG_std = np.std(EEGSignal, axis=1)
    return (EEGSignal - np.array(EEG_mean, ndmin=2).T) / np.array(EEG_std, ndmin=2).T


def commonAverageReference(ds):
    dataset_wo_CAR = []
    for EEGMatrix in ds:
        CAR = np.mean(EEGMatrix, axis=0)
        matrix_wo_CAR = np.subtract(EEGMatrix, CAR)
        dataset_wo_CAR.append(matrix_wo_CAR)
    return dataset_wo_CAR


def main():
    print('Start Training')
    start = timeit.timeit()
    num_trials_train = 1000
    num_trials_test = 300
    if not isDatabaseReady:
        control_patients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        control_patients_train = []
        control_patients_test = []
        for ii in control_patients:
            if ii < 10:
                control_patients_train.append('S0' + str(ii) + 'T.mat')
                control_patients_train.append('S0' + str(ii) + 'E.mat')
                # control_patients_test.append('S0' + str(ii) + 'E.mat') ------------------------------------------------- CAMBIOOOOO!!!!
            else:
                control_patients_train.append('S' + str(ii) + 'T.mat')
                control_patients_test.append('S' + str(ii) + 'E.mat')
        X_train, y_train = loadControlPatients(control_patients_train)
        X_test, y_test = loadControlPatients(control_patients_test)
        # saveDataset(X_train, y_train, X_test, y_test)
    else:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    #show_y_distribution(y_train, y_test)
    #plot_eeg(X_train, X_test)
    # Pick random trials for train and test:
    if not useAllTrial:
        # X_train, y_train = generate_random_dataset(X_train, y_train, num_trials_train)
        X_test, y_test = generate_random_dataset(X_test, y_test, num_trials_test)
    # Vary parameters:
    wd = [1000, 600, 800, 1200, 1400]
    err_param = []
    corr_param = []
    for param in wd:
        X_train_n, y_train_n = generate_random_dataset(X_train, y_train, param)
        train_error_per_epoch, train_correct_per_epoch, eval_error_per_epoch, eval_correct_per_epoch, err_final, corr_final = train_cnn_model(
            X_train_n, y_train_n, X_test, y_test, param)
        err_param.append(err_final)
        corr_param.append(corr_final)
        train_trails_number = param
        test_trails_number = 300
        plot_eval_vs_train(train_error_per_epoch, train_correct_per_epoch, eval_error_per_epoch, eval_correct_per_epoch,
                           train_trails_number, test_trails_number)
    np.save('err_param_trainSet', err_param)
    np.save('corr_param_trainSet', corr_param)
    plot_parameters(err_param,corr_param,wd)
    print('Finished Training')

    end = timeit.timeit()
    print('Time of the process:')
    print(end - start)
    train_trails_number = len(X_train)
    test_trails_number = len(X_test)
    plot_eval_vs_train(train_error_per_epoch, train_correct_per_epoch, eval_error_per_epoch, eval_correct_per_epoch,
                       train_trails_number, test_trails_number)
    # plot_results(train_error_per_batch, correct_per_epoch, test_error_per_batch, correct_test, output_predicted, output_expected)
    print('Process finished.')


def generate_random_dataset(X_in, y_in, num_trials):
    pair_data = list(zip(X_in, y_in))
    pair_data = random.sample(pair_data, num_trials)
    X_out, y_out = zip(*pair_data)
    return list(X_out), np.asarray(y_out)


def train_cnn_model(X_train, y_train, X_test, y_test, param):
    print(param)
    save_path = f'./CNN1d-model-50.pth'
    batch_size = 6
    num_epoch = 200

    # Load data:
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42) #, stratify=True
    params = {'batch_size': batch_size,
              'shuffle': True}

    # Generators
    training_set = Dataset(X_train, y_train)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    test_set = Dataset(X_test, y_test)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    # Model:
    num_classes = 2
    x_axis = 3
    y_axis = 640

    # Create NN:
    # model = Net(num_classes, x_axis, y_axis)
    # model = NetExtended(num_classes, x_axis, y_axis)
    model = NetExtended1D(num_classes, x_axis, y_axis) # best
    # model = ShallowNet(num_classes, x_axis, y_axis)
    # model = DeepConvNet(num_classes, x_axis, y_axis)
    # model = Net2D(num_classes, x_axis, y_axis)

    # Loss Function:
    criterion = nn.CrossEntropyLoss()
    # optimizer = optCNN.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optCNN.Adam(model.parameters(), lr=0.00005, betas=[0.9, 0.95], weight_decay=1e-1)
    # optimizer = optCNN.Adam(model.parameters(), lr=0.0001, betas=[0.9, 0.95])
    lr_decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96) #-------------------------------ADEDED

    train_error_per_epoch = []
    train_correct_per_epoch = []
    eval_error_per_epoch = []
    eval_correct_per_epoch = []
    for epoch in range(num_epoch):
        model, train_error_epoch, train_correct_epoch = train_epoch(training_generator, optimizer, model, criterion)
        eval_error_epoch, eval_correct_epoch = eval_epoch(test_generator, criterion, model)
        print("Epoch:{}/{}; AVG Training Loss:{:.3f}, AVG Evaluation Loss:{:.3f}".format(epoch + 1, num_epoch, np.mean(train_error_epoch), np.mean(eval_error_epoch)))
        train_error_per_epoch.append(np.mean(train_error_epoch))
        train_correct_per_epoch.append(train_correct_epoch)
        eval_error_per_epoch.append(np.mean(eval_error_epoch))
        eval_correct_per_epoch.append(eval_correct_epoch)
        # lr_decay_scheduler.step()  #----------------------------------------------------------------------------------------------ADEDED
    # Saving the model
    # torch.save(model.state_dict(), save_path)

    # Test the model:
    error_final, correct_final = eval_epoch(test_generator, criterion, model)

    return train_error_per_epoch, train_correct_per_epoch, eval_error_per_epoch, eval_correct_per_epoch, np.mean(error_final), correct_final


def train_epoch(training_generator, optimizer, model, criterion):
    model.train()
    error_train = []
    correct = 0
    for inputs, labels in training_generator:
        inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.LongTensor)
        if not is1D:
            inputs = inputs[:, None, :, :]
        optimizer.zero_grad()
        output = model(inputs)
        labels = torch.subtract(labels, 1)
        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        # total correct
        error_train.append(batch_loss.data.detach().numpy())
        preds = np.argmax(output.data.numpy(), axis=-1)
        correct += np.sum(labels.data.numpy() == preds)
    return model, error_train, correct


def eval_epoch(test_generator, criterion, model):
    model.eval()
    error_eval = []
    correct_eval = 0
    for inputs, labels in test_generator:
        inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.LongTensor)
        if not is1D:
            inputs = inputs[:, None, :, :]
        output = model(inputs)  # forward + backward + optimize
        labels = torch.subtract(labels, 1)
        batch_loss = criterion(output, labels)
        error_eval.append(batch_loss.data.detach().numpy())  # test error
        preds = np.argmax(output.data.detach().numpy(), axis=-1)
        correct_eval += np.sum(labels.data.numpy() == preds)
    return error_eval, correct_eval


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


def plot_eval_vs_train(train_error_per_epoch, train_correct_per_epoch, eval_error_per_epoch, eval_correct_per_epoch,
                       train_trails_number, test_trails_number):
    # Print statistics:
    print('Train error, accuracy of the training:')
    print(np.mean(train_error_per_epoch),
          np.sum(train_correct_per_epoch) / (len(train_correct_per_epoch) * train_trails_number))
    print('Eval error, accuracy of the evaluation:')
    print(np.mean(eval_error_per_epoch),
          np.sum(eval_correct_per_epoch) / (len(eval_correct_per_epoch) * test_trails_number))

    # Plot test error per epoch:
    x = np.arange(1, len(train_error_per_epoch) + 1)
    plt.figure(1)
    plt.plot(x, train_error_per_epoch)
    plt.plot(x, eval_error_per_epoch)
    plt.xlabel('Num epochs')
    plt.ylabel('Error')
    plt.title('Error per epoch: Train vs. Evaluation')
    plt.legend(['Train', 'Evaluation'])

    # Plot test error per epoch:
    x = np.arange(1, len(train_correct_per_epoch) + 1)
    plt.figure(2)
    plt.plot(x, np.array(train_correct_per_epoch)/train_trails_number)
    plt.plot(x, np.array(eval_correct_per_epoch)/test_trails_number)
    plt.xlabel('Num epochs')
    plt.ylabel('Correct')
    plt.title('Correct per epoch: Train vs. Evaluation')
    plt.legend(['Train', 'Evaluation'])

    plt.show()


def plot_eeg(X_train, X_test):
    X_used = X_train[0]
    X_used_test = X_test[0]
    fig, axs = plt.subplots(15)
    x = np.arange(0, 5, 1/512)
    fig.suptitle('EEG trial')
    i = 0
    for eeg in X_used:
        axs[i].plot(x, eeg)
        i = i + 1
    i = 0
    for eeg in X_used_test:
        axs[i].plot(x, eeg)
        i = i + 1
    plt.legend(['Train', 'Test'])
    plt.show()


def plot_parameters(err_param, corr_param, wd):
    # WEIGHT DECAY:
    # err_param = [1.483, 1.592, 1.657, 1.107, 0.614, 0.693, 0.693, 0.693]
    # corr_param = [184, 182, 178, 191, 194, 148, 148, 152]
    # xs = [1, 2, 3, 4, 5, 6, 7, 8]
    # labels = ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e1', '1e2']

    # TRAIN SET:
    err_param = [0.69453, 0.64964, 0.60728, 0.59513, 0.58513]
    corr_param = [180, 186, 205, 213, 211]
    # LEARNING RATE:
    #err_param = np.load('err_param_batchSize.npy')
    #corr_param = np.load('corr_param_batchSize.npy')
    xs = [1, 2, 3, 4, 5]
    labels = ['600', '800', '1000', '1200', '1400']


    corr_param = np.divide(corr_param, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.title.set_text('Error varying the train set size')
    ax2.title.set_text('Accuracy varying the train set size')
    fig.suptitle('Train set size influence analysis')
    ax1.plot(xs, err_param)
    plt.xticks(xs, labels)
    plt.setp(ax1, xticks=xs, xticklabels=labels)
    plt.setp(ax2, xticks=xs, xticklabels=labels)
    ax2.plot(xs, corr_param)
    ax1.set_xlabel('Train set size')
    ax2.set_xlabel('Train set size')
    ax1.set_ylabel('Error')
    ax2.set_ylabel('Accuracy')
    plt.show()


def show_y_distribution(y_train, y_test):
    print(len(y_train))
    print(len(y_test))

    plt.figure(1)
    plt.plot(y_train - 1)
    plt.xlabel('Train')

    plt.figure(2)
    plt.plot(y_test - 1)
    plt.xlabel('Traget')

    plt.show()


def saveDataset(X_train, y_train, X_test, y_test):
    # from scipy.io import savemat
    # database = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    # savemat("database.mat", database)
    np.save('X_train', X_train)
    np.save('y_train', y_train)
    np.save('X_test', X_test)
    np.save('y_test', y_test)


def pt_results():
    t_lr5_98 = np.load('train_00005.npy')
    e_lr5_98 = np.load('eval_00005.npy')
    t_lr5 = np.load('train_maxpool.npy')
    e_lr5 = np.load('eval_maxpool.npy')
    t_lr1_98 = np.load('train_00005.npy')
    e_lr1_98 = np.load('eval_00005.npy')


    plt.plot(t_lr5, linestyle=':', color='blue')
    plt.plot(e_lr5, linestyle='-', color='b')
    #plt.plot(t_lr5_98, linestyle=':', color='g')
    #plt.plot(e_lr5_98, linestyle='-', color='g')
    plt.plot(t_lr1_98, linestyle=':', color='r')
    plt.plot(e_lr1_98, linestyle='-', color='r')
    plt.legend(['Train; Max Pooling', 'Eval; Max Pooling', 'Train; Mean Pooling', 'Eval; Mean Pooling'])
    plt.xlabel('Num epochs')
    plt.ylabel('Error')
    plt.title('Pooling layer comparison')
    plt.show()


#main()
# plot_parameters(1, 1, 1)
pt_results()

