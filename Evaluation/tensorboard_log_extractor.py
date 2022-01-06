import numpy
from tensorflow.python.summary.summary_iterator import summary_iterator
from google.protobuf.json_format import MessageToJson
import numpy as np

i = 0


def extractor(path_train, path_test, dim=4):
    train1 = summary_iterator(path_train)
    test1 = summary_iterator(path_test)
    '''
    print(type(s))

    d = iter(s)
    print(type(d))
    t= next(d)
    print(t)'''

    data = np.zeros([50, dim])
    for e in train1:
        # print(e.step)
        for v in e.summary.value:
            if v.tag == 'epoch_accuracy':
                data[e.step, 0] = v.simple_value
            elif v.tag == 'epoch_loss':
                data[e.step, 1] = v.simple_value
            else:
                pass

    for e in test1:
        # print(e.step)
        for v in e.summary.value:
            if v.tag == 'epoch_accuracy':
                data[e.step, 2] = v.simple_value
            elif v.tag == 'epoch_loss':
                data[e.step, 3] = v.simple_value
            else:
                pass

    return data


path_train1 = '../Train/logs/fit/20220106-123415/train/events.out.tfevents.1641490455.pedram-Z97M-DS3H.22914.625.v2'
path_test1 = '../Train/logs/fit/20220106-123415/validation/events.out.tfevents.1641490459.pedram-Z97M-DS3H.22914.3729.v2'
data1 = extractor(path_train1, path_test1)
numpy.savetxt('model1.csv', data1, delimiter=',')
print('first done')
# print(dtrain1)
# print('hi')
# print(dtest1)


path_train2 = '../Train/logs/fit/20220106-123122/train/events.out.tfevents.1641490282.pedram-Z97M-DS3H.22550.625.v2'
path_test2 = '../Train/logs/fit/20220106-123122/validation/events.out.tfevents.1641490287.pedram-Z97M-DS3H.22550.3717.v2'
data2 = extractor(path_train2, path_test2)
numpy.savetxt('model2.csv', data2, delimiter=',')
print('second done')
# print(dtrain3)
# print('hi')
# print(dtest3)
