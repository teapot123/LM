import argparse
import datasets

def parse_conf(conf):
    try:
        if ' ' in conf:
            conf = conf.split(' ')[0]
        if conf[-1] == '.':
            conf = conf[:-1]
        conf = eval(conf)
    except SyntaxError:
        print(f"Cannot parse confidence: {conf}")
        return None
    except IndexError:
        print(f"Cannot parse confidence: {conf}")
        return None
    except NameError:
        print(f"Cannot parse confidence: {conf}")
        return None
    except TypeError:
        print(f"Cannot parse confidence: {conf}")
        return None
    return conf

def read_res_file(filename, first_num):
    res_dict = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            if first_num != None and i >= first_num:
                continue
            tmp = line.split('\t')
            answers = []
            confs = []
            if len(tmp) == 3:
                question = tmp[0].strip()
                answer = tmp[1].strip()
                conf = tmp[2].strip()
                conf = parse_conf(conf)
                if conf != None:
                    answers.append(answer)
                    confs.append(conf)
            else:
                ind = 1
                question = tmp[0].strip()
                print('\t'.join(tmp[1:]))
                for t in tmp[1:]:
                    if len(t.strip()) == 0:
                        continue
                    if f'G{ind}' in t:
                        tok = f'G{ind}:'
                        answer = t.split(tok)[1]
                        next_tok = f'P{ind}:'
                        answer = answer.split(next_tok)[0].strip().split('(')[0].strip()    
                    if f'P{ind}' in t:
                        tok = f'P{ind}:'
                        next_tok = f'G{ind+1}:'
                        conf = t.split(tok)[1]
                        conf = conf.split(next_tok)[0].strip().split('(')[0].split(')')[0].strip()
                        conf = parse_conf(conf)
                        if conf != None:
                            answers.append(answer)
                            confs.append(conf)
                            print(f"answer: {answer} conf: {conf}")
                        ind += 1
            if answers != []:
                res_dict[question] = [answers, confs]
            else:
                continue

    return res_dict

def read_dataset(filename):
    val_data = datasets.load_from_disk(filename)
    gt_dict = {}
    for data in val_data:
        # if data['question'] in gt_dict:
        #     question = data['question']
        #     print(f'repeat questions: {question}')
        gt_dict[data['question']] = data['answer']
    return gt_dict

def eval_word_match(res_data, gt_data, bin_num):
    conf_list = [{'correct':0, 'total':0, 'conf':0} for x in range(bin_num)]
    correct = 0
    total = 0
    for question in gt_data:
        if question not in res_data:
            # print(f'question not in results: {question}')
            continue
        gt = gt_data[question]
        answers, confs = res_data[question]
        for (res, conf) in zip(answers, confs):
            conf_index = int(conf * bin_num)
            if conf_index == bin_num:
                conf_index = bin_num - 1
            # exact word match
            gt_list = list(set([g.lower() for g in gt]))
            if res.lower() in gt_list:
                correct += 1
                conf_list[conf_index]['correct'] += 1
            else:
                pass
                # print(f'Question: {question}')
                # print(f'Incorrect answer: {res}  Ground Truth: {gt}')
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
        print(f'ground truth num: {len(gt_data)}')
        eval_word_match(res_data, gt_data, args.bin)