from model.DEAP_model import *
from utils import timer, visual_loss
from data.DEAP.data_loader_1D import *
import torch.utils.data as Data
from  model.Regular import Regularization

#args
fold_cross_num = 10
pre_process = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_class = 2
num_epochs = 300
batch_size = 300
learning_rate = 0.1
weight_decay=0.001


def record_test(data_loader):
    test_record = {'n0': 0, 'n1': 0}
    for i in data_loader.testLabel:
        if i == 0:
            test_record['n0'] += 1
        elif i == 1:
            test_record['n1'] += 1
    return test_record

def cal_accu(y_pred:'list', y_label:'list'):
    y_pred = np.argmax(y_pred,axis=-1)
    acc = np.equal(y_label, y_pred)
    acc = np.mean(acc)
    return acc, y_label, y_pred


def predict(model, data, label, total, correct):
    n0_true = 0
    n0_false = 0
    n1_true = 0
    n1_false = 0

    data = data.to(device)
    label = label.to(device)
    outputs = model(data)
    acc, lab_list, pre_list = cal_accu(outputs.tolist(), label.tolist())
    total += len(lab_list)
    correct += acc * len(lab_list)
    print('pred:%s' % pre_list)
    print('label:%s' % lab_list)
    for i in range(len(pre_list)):
        if lab_list[i] == 0:
            if pre_list[i] == lab_list[i]:
                n0_true += 1
            else:
                n0_false += 1
        else:
            if pre_list[i] == lab_list[i]:
                n1_true += 1
            else:
                n1_false += 1
    return total, correct, n0_true, n0_false, n1_true, n1_false

def train(model, criterion, optimizer, data, label, reg_loss):
    data = data.to(device)
    label = label.to(device)

    # forward pass
    outputs = model(data)
    loss = criterion(outputs, label)
    # if weight_decay > 0:
    #     loss = loss + reg_loss(model)

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), outputs.cpu()

def run_train(type, cvx, use_attention = True, use_non_local = True, load_model = False):
    #create data_loader
    data_loader = data_loader_1D(type, cvx, fold_cross_num)

    Record = {'N0_true': 0, 'N0_false': 0, 'N1_true': 0, 'N1_false': 0}
    Test_accuracy = []

    # for fold_num in range(fold_cross_num):
    for fold_num in range(1):
        model_name = 'DEAP_Resnet1d_' + type + '_'
        if use_attention: model_name += 'A'
        if use_non_local: model_name += 'L'

        model_name = model_name + str(fold_num)
        model = Model(use_attention, use_non_local, num_class=2)
        if load_model:
            model.load_state_dict(torch.load('./save/' + model_name + '.pt'))
            model_name = 'DEAPcontinue_Resnet1d_' + type + '_'
            if use_attention: model_name += 'A'
            if use_non_local: model_name += 'L'
            model = model.to(device)
        else:
            model = model.to(device)
        print(model)
        visualizer_env = model_name + str(fold_num)
        visualizer = visual_loss.Visualizer(env=visualizer_env)
        print('train fold %s' % fold_num)
        data_loader.fold_arrange_id(fold_num)
        test_record = record_test(data_loader)
        record = {'n0_true': 0, 'n0_false': 0, 'n1_true': 0, 'n1_false': 0}
        loss_list = []
        train_accuracies = []

        train_dataId_tensor = torch.Tensor(np.array(data_loader.trainId))
        train_label_tensor = torch.Tensor(np.array(data_loader.trainLabel))

        test_dataId_tensor = torch.Tensor(np.array(data_loader.testID))
        test_label_tensor = torch.Tensor(np.array(data_loader.testLabel))

        train_set = Data.TensorDataset(train_dataId_tensor, train_label_tensor)
        test_set = Data.TensorDataset(test_dataId_tensor, test_label_tensor)

        train_loader = Data.DataLoader(dataset=train_set,
                                       batch_size=batch_size,
                                       shuffle=True
                                       )
        test_loader = Data.DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      shuffle=True
                                      )
        if weight_decay > 0:
            reg_loss = Regularization(model, weight_decay, p=2).to(device)
        else:
            print("no regularization")
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.001)

        train_total_step = len(train_loader)
        test_total_step = len(test_loader)

        for epoch in range(num_epochs):
            train_correct = 0
            train_total = 0
            for batch_idx, (dataId, target) in enumerate(train_loader):
                target = target.long()
                model.train()
                model.zero_grad()
                print('Epoch [{}/{}],Step[{}/{}]'.format(epoch + 1, num_epochs, batch_idx + 1, train_total_step))
                dataId = np.array(dataId, dtype=np.int).tolist()
                data = torch.Tensor(get_batch_data(dataId, data_loader))
                loss, outputs = train(model, criterion, optimizer, data, target, reg_loss)
                # if epoch == 10:
                #     optimizer.defaults['lr'] = learning_rate * 0.1
                loss_list.append(loss)
                print('Loss:{}'.format(loss))
                visualizer.plot_many_stack({visualizer_env + '_loss': loss})
                train_accuracy, label,pred = cal_accu(outputs.tolist(),target.tolist())
                train_correct += train_accuracy*len(label)
                train_total += len(label)

            train_accuracies.append(train_correct/train_total )
            visualizer.plot_many_stack({visualizer_env + '_train accuracy': train_accuracy})

            # save loss and train accuracy each epoch
            with open('./save/loss/' + visualizer_env + '_loss.txt', 'w') as f:
                f.write(str(loss_list))
            with open('./save/loss/' + visualizer_env + '_train accuracy.txt', 'w') as f:
                f.write(str(train_accuracies))

            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                for batch_idx, (dataId, target) in enumerate(test_loader):
                    print('test step:[{}/{}]\n'.format(batch_idx + 1, test_total_step))
                    dataId = np.array(dataId, dtype=np.int).tolist()
                    data = torch.Tensor(get_batch_data(dataId, data_loader))
                    total, correct, a, b, c, d = predict(model, data, target, total, correct)
                    record['n0_true'] += a
                    record['n0_false'] += b
                    record['n1_true'] += c
                    record['n1_false'] += d
                test_accuracy = 100 * correct / total
                visualizer.plot_many_stack({visualizer_env + '_test accuracy': test_accuracy})
                print('Test Accuracy of the model is {}%'.format(test_accuracy))
                Test_accuracy.append(test_accuracy)
                file_name = './save/' + model_name + '.pt'
                if test_accuracy > 70:
                    torch.save(model.state_dict(), file_name)

    # print(Test_accuracy)
    Test_accuracy = np.array((Test_accuracy))
    print('accu = %s' % Test_accuracy.mean())

if __name__ == '__main__':
    run_train('Arousal','tonic', use_attention = True, use_non_local = True)
    run_train('Valence','tonic', use_attention = True, use_non_local = True)
