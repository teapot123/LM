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

def read_res_file(filename, first_num, guess_num, read_conf):
    res_dict = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            if first_num != None and i >= first_num:
                continue
            tmp = line.strip().split('\t')
            answers = []
            confs = []
            if guess_num == 1 and not read_conf:
                confs.append(1)
                if len(tmp) == 1:
                    print(line)
                    try:
                        answer = ' '.join(line.split('?')[1].split(' ')[1:]).strip()
                        question = line.split(answer)[0].strip()
                    except IndexError:
                        continue
                    except ValueError:
                        continue
                else:
                    question = tmp[0].strip()
                    for t in tmp[1:]:
                        if len(t.strip()) == 0:
                            continue
                        if 'Guess:' in t:
                            answer = t.split('Guess:')[1].strip()
                        elif ':' in t:
                            answer = t.split(':')[1].strip()
                        else:
                            answer = t.strip()
                        break
                if 'The question is: ' in question:
                    question = question.split('The question is: ')[1].strip()
                if '</s>' in answer:
                    answer = answer.split('</s>')[0]
                answer = answer.replace('<', '').replace('>', '')
                answers.append(answer)
                # print(question)
                # print(answers)
            elif guess_num == 1 and read_conf:
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
        gt_dict[data['question']] = data['answers']
    return gt_dict

def eval_word_match(res_data, gt_data, bin_num, read_conf):
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
            print(f'Question: {question}')
            if res.lower() in gt_list:
                res_dict[question]['correct'] = 1
                print(f'Correct answer: {res} ')
            else:
                res_dict[question]['correct'] = 0
                # print(f'Question: {question}')
                print(f'Incorrect answer: {res} ')
            res_dict[question]['conf'] = conf
            break

    total = len(res_dict)
    # print(res_dict)
    correct = sum([res_dict[q]['correct'] for q in res_dict])
    # print(correct)

    if read_conf:
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
    if read_conf:
        print(f'ece: {ece}')


def eval_word_match_multiple_answer(res_data, gt_data, question_match_file):
    res_dict = {}
    last_question = ''
    i = 0
    
    new2old = {}
    old2new = {}
    with open(question_match_file) as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            if len(tmp) != 2:
                continue
            orig_q = tmp[0]
            new_q = tmp[1]
            if orig_q == new_q:
                continue
            new2old[new_q] = orig_q
            if orig_q not in old2new:
                old2new[orig_q] = set()
            old2new[orig_q].add(new_q)

    res_dict = {}
    for question in gt_data:
        gt = gt_data[question]
        res_question = [x for x in res_data if question in x]
        res_question = list(set(res_question))
        answer0, conf0 = [], []
        answers, confs = [], []
        gt_list = list(set([g.lower() for g in gt]))
        if len(res_question) > 0:
            answer0, conf0 = res_data[res_question[0]]
            print(f'Original Question: {res_question[0]}')
            res_dict[question] = {'orig':0,}
            res = answer0[0]
            if res.lower() in gt_list:
                res_dict[question]['orig'] = 1
                print(f'Correct answer: {res} ')
            else:
                res_dict[question]['orig'] = 0
                # print(f'Question: {question}')
                print(f'Incorrect answer: {res} ')
        # else:
        #     print(f'no original question: {question}')
        #     continue
        res_question = [x for x in old2new if question in x]
        if len(res_question) > 0:
            new_questions = old2new[res_question[0]]
            for q in new_questions:
                question_in_file = [x for x in res_data if q in x]
                if len(question_in_file) > 0:
                    print(f'New Question: {question_in_file[0]}')
                    res = res_data[question_in_file[0]][0][0]
                    if question not in res_dict:
                        res_dict[question] = {'new_correct':0, 'new_incorrect':0, 'correct_num':0, 'incorrect_num':0}
                    if 'new_correct' not in res_dict[question]:
                        res_dict[question]['new_correct'] = 0
                        res_dict[question]['correct_num'] = 0
                    if 'new_incorrect' not in res_dict[question]:
                        res_dict[question]['new_incorrect'] = 0
                        res_dict[question]['incorrect_num'] = 0
                    if res.lower() in gt_list:
                        res_dict[question]['new_correct'] = 1
                        res_dict[question]['correct_num'] += 1
                        print(f'Correct answer: {res} ')
                    else:
                        res_dict[question]['new_incorrect'] = 1
                        res_dict[question]['incorrect_num'] += 1
                        # print(f'Question: {question}')
                        print(f'Incorrect answer: {res} ')
                    # res_dict[question]['conf'] = conf
                    answers.extend(res_data[question_in_file[0]][0])
                    confs.extend(res_data[question_in_file[0]][1])
                # else:
                #     print(f'no new question: {q}')
                #     continue
        if question in res_dict:
            if 'orig' in res_dict[question] and 'new_correct' in res_dict[question]:
                if res_dict[question]['orig'] != res_dict[question]['new_correct']:
                    print('----------Accuracy Changed!--------')
                    print(res_dict[question])
                elif res_dict[question]['orig'] == res_dict[question]['new_incorrect']:
                    print('----------Accuracy Changed!--------')
                    print(res_dict[question])

            
    overlaps = [x for x in res_dict if 'orig' in res_dict[x] and ('new_correct' in res_dict[x] or 'new_incorrect' in res_dict[x])]
    total = len(overlaps)
    # print(res_dict)
    correct = sum([res_dict[q]['orig'] for q in overlaps])
    # print(correct)
    max_correct = sum([res_dict[q]['new_correct'] or res_dict[q]['orig'] for q in overlaps])/total
    min_correct = 1 - sum([res_dict[q]['new_incorrect'] or (not res_dict[q]['orig']) for q in overlaps])/total
    consist_correct = sum([(res_dict[q]['correct_num'] + (1.1 if res_dict[q]['orig']==1 else -1.1) > res_dict[q]['incorrect_num']) for q in overlaps])/total

    print(f'acc: {correct/total} total: {total}')
    print(f'min acc: {min_correct}   max acc: {max_correct}')
    print(f'consist acc: {consist_correct}')



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--res_file', type=str)
    parser.add_argument('--question_match_file', type=str, default=None)
    parser.add_argument('--bin', type=int, default=20)
    parser.add_argument('--first_num', type=int, default=None)
    parser.add_argument('--guess_num', type=int, default=1)
    parser.add_argument('--conf', type=bool, default=False)
    args = parser.parse_args()
    res_data = read_res_file(args.res_file, args.first_num, args.guess_num, args.conf)
    # print(res_data)

    if args.dataset == 'trivia_qa':
        gt_data = read_dataset('data/trivia_qa/validation_1000')
        print(f'ground truth num: {len(gt_data)}')
        # print(f'gt dict: {gt_data}')
        if args.question_match_file is not None:
            eval_word_match_multiple_answer(res_data, gt_data, args.question_match_file)
        else:
            eval_word_match(res_data, gt_data, args.bin, args.conf)