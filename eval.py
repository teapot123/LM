import argparse
import datasets

def parse_conf(conf):
    try:
        conf = conf.replace('(', '').replace(')', '').replace(',','')
        if ' ' in conf:
            conf = conf.split(' ')[0]
        if conf[-1] == '.':
            conf = conf[:-1]
        if conf[-1] == '%':
            conf = conf[:-1] + '/100'
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
            if 'Guess:' in line and 'Probability:' in line:
                question = tmp[0].strip()
                for t in tmp[1:]:
                    if len(t.strip()) == 0:
                        continue
                    if 'Guess:' in t:
                        answer = t.split('Guess:')[1].strip()
                    elif 'Probability:' in t:
                        conf = t.split('Probability:')[1].strip()
                conf = parse_conf(conf)
                if conf != None:
                    answers.append(answer)
                    confs.append(conf)
            else:
                ind = 1
                question = tmp[0].strip()
                # print('\t'.join(tmp[1:]))
                for t in tmp[1:]:
                    if len(t.strip()) == 0:
                        continue
                    if f'G{ind}: ' in t:
                        tok = f'G{ind}: '
                        answer = t.split(tok)[1]
                        next_tok = f'P{ind}: '
                        answer = answer.split(next_tok)[0].strip().split('(')[0].strip()    
                    if f'P{ind}: ' in t:
                        tok = f'P{ind}: '
                        next_tok = f'G{ind+1}: '
                        conf = t.split(tok)[1]
                        conf = conf.split(next_tok)[0].strip().split('(')[0].split(')')[0].strip()
                        conf = parse_conf(conf)
                        if conf != None:
                            answers.append(answer)
                            confs.append(conf)
                            # print(f"answer: {answer} conf: {conf}")
                        ind += 1
                print(f"conf: {confs}")
                sum_conf = sum(confs)
                confs = [x/sum_conf for x in confs]
                print(f"conf: {confs}")
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
    res_dict = {}
    last_question = ''
    i = 0

    for question in gt_data:
        gt = gt_data[question]
        if question not in res_data:
            res_question = [x for x in res_data if question in x]
            res_question = list(set(res_question))
            if len(res_question) == 0:
                # print(f'question not in results: {question}')
                continue
            else:
                answers, confs = res_data[res_question[0]] 
            # else:
            #     print(f'questions in results: {res_question}')
            #     continue
        else: 
            answers, confs = res_data[question]
        for (res, conf) in zip(answers, confs):
            # exact word match
            gt_list = list(set([g.lower() for g in gt]))
            res_dict[question] = {}
            if res.lower() in gt_list:
                res_dict[question]['correct'] = 1
                # print(f'Correct answer: {res}  Ground Truth: {gt}')
            else:
                res_dict[question]['correct'] = 0
                # print(f'Question: {question}')
                # print(f'Incorrect answer: {res}  Ground Truth: {gt}')
            res_dict[question]['conf'] = conf
            break

    total = len(res_dict)
    # print(res_dict)
    correct = sum([res_dict[q]['correct'] for q in res_dict])
    # print(correct)

    ece = 0
    conf_list = sorted(res_dict.items(), key = lambda x:x[1]['conf'])
    # print(conf_list)
    bin_weight = total / bin_num
    for i in range(bin_num):
        start_index, end_index = int(bin_weight * i), min(int(bin_weight * (i+1)), total - 1)
        weight = end_index - start_index
        bin_list = conf_list[start_index:end_index]
        avg_acc = sum([x[1]['correct'] for x in bin_list]) / weight
        avg_conf = sum([x[1]['conf'] for x in bin_list]) / weight
        ece += weight * abs(avg_acc - avg_conf)
        print(f"start: {start_index} {bin_list[0][1]['conf']} end: {end_index} {bin_list[-1][1]['conf']}")
        print(f"avg conf: {avg_conf} avg acc: {avg_acc}")
    ece /= total
    
    print(f'acc: {correct/total} total: {total}')
    # print(f'conf list: {res_dict}')
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