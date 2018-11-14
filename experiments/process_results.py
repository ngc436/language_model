def process_file(negative_path, positive_path):
    flag = False
    false_negative = 0
    true_negative = 0
    with open(negative_path) as f:
        for line in f:
            if line.startswith('=='):
                flag = True
                continue
            if flag is True:
                if line.strip() == '-':
                    false_negative += 1
                    flag = False
                    continue
                if line.strip() == '+':
                    true_negative += 1
                    flag = False
                    continue
                break
    print('FN: %s' % false_negative)
    print('TN: %s' % true_negative)
    flag = False
    false_positive = 0
    true_positive = 0
    with open(positive_path) as f:
        for line in f:
            if line.startswith('=='):
                flag = True
                continue
            if flag is True:
                if line.strip() == '-':
                    false_positive += 1
                    flag = False
                    continue
                if line.strip() == '+':
                    true_positive += 1
                    flag = False
                    continue
                break
    print('FP: %s' % false_positive)
    print('TP: %s' % true_positive)
    recall = true_positive / (true_positive + false_positive)
    precision = true_positive / (true_positive + false_negative)
    print('=====  Class 1  =====')
    print('Precision: %s, Recall: %s, F1: %s' % (recall, precision, 2 * (recall * precision) / (recall + precision)))

    print()

    recall = true_negative / (true_negative + false_negative)
    precision = true_negative / (true_negative + false_positive)
    print('=====  Class 0  =====')
    print('Precision: %s, Recall: %s, F1: %s' % (recall, precision, 2 * (recall * precision) / (recall + precision)))


def extract_for_verification(positive_path, negative_path):
    raise NotImplementedError


def main():
    positive_path = 'results/positive_comments_big_marked.txt'
    negative_path = 'results/negative_comments_big_marked.txt'
    process_file(negative_path, positive_path)


if __name__ == '__main__':
    main()
