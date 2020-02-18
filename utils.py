import numpy as np

from consts import NONE, PAD


def build_vocab(labels, BIO_tagging=True):
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label


def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1
    # print(y_true)
    # print(y_pred)
    # print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]
# def find_triggers(labels):
#     """
#     :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
#     :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
#     """
#     result = []
#     labels = [label for label in labels]
#
#     for i in range(len(labels)):
#         if (labels[i]!="O") & (labels[i]!="[PAD]"):
#             if i == 0:
#                 result.append([i, i + 1, labels[i]])
#             else:
#                 if labels[i] != labels[i-1]:
#                     result.append([i, i + 1, labels[i]])
#
#     for i in range(len(result)):
#         item = result[i]
#         j = item[1]
#         while j < len(labels):
#             if labels[j] == labels[j-1]:
#                 j = j + 1
#                 item[1] = j
#             else:
#                 break
#         result[i] = tuple(item)
#     return result

def find_argument(labels):
    result = []

    labels = [label for label in np.int64(labels)]
    for i in range(len(labels)):
        if labels[i]>1:
            if i == 0:
                result.append([i, i + 1, labels[i]])
            else:
                if labels[i] != labels[i-1]:
                    result.append([i, i + 1, labels[i]])

    for i in range(len(result)):
        item = result[i]
        j = item[1]
        while j < len(labels):
            if labels[j] == labels[j-1]:
                j = j + 1
                item[1] = j
            else:
                break
        result[i] = tuple(item)
    return result




# To watch performance comfortably on a telegram when training for a long time
def report_to_telegram(text, bot_token, chat_id):
    try:
        import requests
        requests.get('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(bot_token, chat_id, text))
    except Exception as e:
        print(e)
