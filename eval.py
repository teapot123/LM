import argparse
import datasets

def read_res_file(filename, first_num):
    res_dict = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            if first_num != None and i >= first_num:
                continue
            tmp = line.split('\t')
            if len(tmp) != 3:
                continue
            question = tmp[0].strip()
            answer = tmp[1].strip()
            conf = tmp[2].strip()
            try:
                if conf[-1] == '.':
                    conf = conf[:-1]
                if ' ' in conf:
                    conf = conf.split(' ')[0]
                conf = eval(conf)
            except SyntaxError:
                print(f"Cannot parse confidence: {conf}")
                continue
            res_dict[question] = [answer, conf]

    return res_dict

def read_dataset(filename):
    val_data = datasets.load_from_disk(filename)
    gt_dict = {}
    for data in val_data:
        gt_dict[data['question']] = data['answer']
    return gt_dict

def eval_word_match(res_data, gt_data, bin_num):
    conf_list = [{'correct':0, 'total':0, 'conf':0} for x in range(bin_num)]
    correct = 0
    total = 0
    for question in gt_data:
        if question not in res_data:
            print(f'question not in results: {question}')
            continue
        gt = gt_data[question]
        res, conf = res_data[question]
        conf_index = int(conf * bin_num)
        if conf_index == bin_num:
            conf_index = bin_num - 1
        if gt.lower() == res.lower():
            correct += 1
            conf_list[conf_index]['correct'] += 1
        else:
            print(f'Question: {question}')
            print(f'Incorrect answer: {res}  Ground Truth: {gt}')
        total += 1
        conf_list[conf_index]['conf'] += conf
        conf_list[conf_index]['total'] += 1
    ece = 0
    for i, bin in enumerate(conf_list):
        if bin['total'] == 0:
            continue
        else:
            avg_acc = bin['correct']/bin['total']
            avg_conf = bin['conf']/bin['total']
            ece += bin['total'] * abs(avg_acc - avg_conf)
    ece /= total
    
    print(f'acc: {correct/total} total: {total}')
    print(f'conf list: {conf_list}')
    print(f'ece: {ece}')


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--res_file', type=str)
    parser.add_argument('--bin', type=int, default=20)
    parser.add_argument('--first_num', type=int, default=None)
    args = parser.parse_args()
    res_data = read_res_file(args.res_file, args.first_num)

    if args.dataset == 'trivia_qa':
        gt_data = read_dataset('data/trivia_qa/validation_1000')
        eval_word_match(res_data, gt_data, args.bin)