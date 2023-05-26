import ipdb
import json
import pickle


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    # if "rest1456" in data_path:
    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)
    #         for ins in data['data']:
    #             sents.append(ins['source'])
    #             labels.append(ins['target'])
    #     print(f"Total examples = {len(sents)}")
    #     return sents, labels
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                words, tuples = words.split(), eval(tuples)
                # 在这里默认按照aspect在文本中出现的顺序进行排序，如果aspect相同则按照opinion出现的顺序进行排序
                tuples.sort(key=lambda x: (x[0][-1], x[1][-1]))
                sents.append(words)
                labels.append(tuples)

    return sents, labels


def read_shot_ratio_from_file(data_path):
    # ipdb.set_trace()
    sents, labels = [], []
    transfer = {'positive': 'POS', 'negative': 'NEG', 'neutral': "NEU"}
    with open(data_path, encoding='UTF-8') as fp:
        for line in fp:
            line = json.loads(line)
            tuples = []
            for t in line['relation']:
                sentiment = transfer[t['type']]
                aspect, opinion = None, None
                for ao in t['args']:
                    if ao['type'] == 'aspect':
                        aspect = ao['offset']
                    if ao['type'] == 'opinion':
                        opinion = ao['offset']
                tuples.append((aspect, opinion, sentiment))
            tuples.sort(key=lambda x: (x[0][-1], x[1][-1]))
            sents.append(line['tokens'])
            labels.append(tuples)
    return sents, labels


if __name__ == "__main__":
    data_path = "/home/zhoushen/ABSA/ABSAData/ASTE/rest1456/train.txt"
    read_line_examples_from_file(data_path)
