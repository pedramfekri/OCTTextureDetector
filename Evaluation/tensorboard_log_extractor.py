import numpy
from tensorflow.python.summary.summary_iterator import summary_iterator
from google.protobuf.json_format import MessageToJson
import numpy as np

i = 0


def extractor(path_train, path_test, dim):
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
                data[e.step, 3] = v.simple_value
            elif v.tag == 'epoch_loss':
                data[e.step, 4] = v.simple_value
            else:
                pass

    return data


path_train1 = '../Train/logs/fit/20211201-202144/train/events.out.tfevents.1638408105.pedram-Z97M-DS3H.49764.15512.v2'
path_test1 = '../Train/logs/fit/20211201-202144/validation/events.out.tfevents.1638408311.pedram-Z97M-DS3H.49764.3208897.v2'
data1 = extractor(path_train1, path_test1)
numpy.savetxt('model1.csv', data1, delimiter=',')
print('first done')
# print(dtrain1)
# print('hi')
# print(dtest1)

path_train2 = '../Train/logs/fit/20211201-141242/train/events.out.tfevents.1638385963.pedram-Z97M-DS3H.47692.15960.v2'
path_test2 = '../Train/logs/fit/20211201-141242/validation/events.out.tfevents.1638386174.pedram-Z97M-DS3H.47692.3210637.v2'
data2 = extractor(path_train2, path_test2)
numpy.savetxt('model2.csv', data2, delimiter=',')
print('second done')
# print(dtrain2)
# print('hi')
# print(dtest2)

path_train3 = '../Train/logs/fit/20211201-085152/train/events.out.tfevents.1638366713.pedram-Z97M-DS3H.43956.15736.v2'
path_test3 = '../Train/logs/fit/20211201-085152/validation/events.out.tfevents.1638366927.pedram-Z97M-DS3H.43956.3209767.v2'
data3 = extractor(path_train3, path_test3)
numpy.savetxt('model3.csv', data3, delimiter=',')
print('third done')
# print(dtrain3)
# print('hi')
# print(dtest3)

path_train4 = '../Train/logs/fit/20211201-000721/train/events.out.tfevents.1638335242.pedram-Z97M-DS3H.38303.592.v2'
path_test4 = '../Train/logs/fit/20211201-000721/validation/events.out.tfevents.1638335325.pedram-Z97M-DS3H.38303.3223.v2'
data4 = extractor(path_train4, path_test4)
numpy.savetxt('model4.csv', data4, delimiter=',')
print('fourth done')

path_train5 = '../Train/logs/fit/20211130-203724/train/events.out.tfevents.1638322645.pedram-Z97M-DS3H.36819.816.v2'
path_test5 = '../Train/logs/fit/20211130-203724/validation/events.out.tfevents.1638322729.pedram-Z97M-DS3H.36819.4093.v2'
data5 = extractor(path_train5, path_test5)
numpy.savetxt('model5.csv', data5, delimiter=',')
print('fifth done')