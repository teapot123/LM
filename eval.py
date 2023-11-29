import argparse
import datasets
from sklearn.metrics import roc_auc_score
import random

from data_utils import read_res_file, read_dataset

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
            # print(f'Question: {question}')
            if res.lower() in gt_list:
                res_dict[question]['correct'] = 1
                # print(f'Correct answer: {res} ')
            else:
                res_dict[question]['correct'] = 0
                # print(f'Question: {question}')
                # print(f'Incorrect answer: {res} ')
            res_dict[question]['conf'] = conf + random.uniform(-1e-3,1e-3)
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
        auroc = 0
        res_dict_items = res_dict.items()
        y_true = [x[1]['correct'] for x in res_dict_items]
        y_score = [x[1]['conf'] for x in res_dict_items]
        auroc = roc_auc_score(y_true, y_score)

        
    
    print(f'acc: {correct/total} total: {total}')
    # print(f'conf list: {res_dict}')
    if read_conf:
        print(f'ece: {ece} auroc: {auroc}')


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
            # print(f'Original Question: {res_question[0]}')
            res_dict[question] = {'orig':0,}
            res = answer0[0]
            if res.lower() in gt_list:
                res_dict[question]['orig'] = 1
                # print(f'Correct answer: {res} ')
            else:
                res_dict[question]['orig'] = 0
                # print(f'Incorrect answer: {res} ')
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