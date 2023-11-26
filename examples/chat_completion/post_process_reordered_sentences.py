import argparse
import datasets


def read_dataset(filename):
    val_data = datasets.load_from_disk(filename)
    gt_dict = {}
    for data in val_data:
        # if data['question'] in gt_dict:
        #     question = data['question']
        #     print(f'repeat questions: {question}')
        gt_dict[data['question']] = data['answers']
    return gt_dict

def match_reordered_questions(gt_data, input_file):
    output_data = {}
    with open(input_file) as f:
        input_data = f.read().strip()
    start_prompt = '<s>Reorganize the question "'
    middle_prompt = '" by reordering the phrases "'
    end_prompt = ' into the following order: '
    # lines = input_data.split(start_prompt)
    lines = input_data.split('\n')
    for new_line in lines:
        tmp = new_line.split('\t')
        for i,line in enumerate(tmp):
            if i == 0:
                if start_prompt not in line or middle_prompt not in line:
                    print(line)
                original_question = line.split(start_prompt)[1].split(middle_prompt)[0]
                noun_phrases = line.split(end_prompt)[0].split(middle_prompt)[-1]
                noun_phrases_len = len(noun_phrases.split(' '))
                if original_question not in output_data:
                    if original_question in gt_data:
                        output_data[original_question] = set()
                    else:
                        print(f"Question {original_question} not found!")
                        continue
                continue
            new_question = line
            if '</s>' in new_question:
                new_question = new_question.split('</s>')[0].strip()
            if ':' in new_question:
                mark = new_question.split(':')[0].strip().lower()
                if 'explanation' in mark:
                    continue
                new_question = new_question.split(':')[-1].strip()
            if len(new_question) == 0:
                continue
            if new_question[0] == '"':
                new_question = new_question[1:]
            if new_question[-1] == '"':
                new_question = new_question[:-1]
            if new_question[-1] != '?':
                print("Not a question!")
                print(new_question)
                continue
            if new_question == original_question:
                continue
            new_q_len = len(new_question.split(' '))
            orig_q_len = len(original_question.split(' '))
            if orig_q_len - new_q_len > new_q_len - noun_phrases_len:
                print("Not a question!")
                print(new_question)
            else:
                output_data[original_question].add(new_question)
    return output_data
        

def match_modified_questions(gt_data, input_file):
    output_data = {}
    with open(input_file) as f:
        input_data = f.read().strip()
    start_prompt = 'Slightly modify Question 2 to have similar meaning with Question 1 but different wordings:'
    lines = input_data.split('\n')
    for new_line in lines:
        original_question = new_line.split("<s>Question 1: ")[1].split('\t')[0].strip()
        if original_question not in output_data:
            if original_question in gt_data:
                output_data[original_question] = set()
            else:
                print(f"Question {original_question} not found!")
                continue
        tmp = new_line.split(start_prompt)[1]
        tmp = tmp.split('\t')
        for line in tmp:
            new_question = line.strip()
            if len(new_question) == 0:
                continue
            if '</s>' in new_question:
                new_question = new_question.split('</s>')[0].strip()
            if ':' in new_question:
                new_question = new_question.split(':')[-1].strip()
            if len(new_question) == 0:
                continue
            if new_question[0] == '"':
                new_question = new_question[1:]
            if new_question[-1] == '"':
                new_question = new_question[:-1]
            if new_question[-1] != '?':
                print("Not a question!")
                print(new_question)
                continue
            if new_question == original_question:
                continue
            output_data[original_question].add(new_question)
    return output_data


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    if args.dataset == 'trivia_qa':
        gt_data = read_dataset('data/trivia_qa/validation_1000')
        # gt_data = read_dataset('data/trivia_qa/train_10000')
        print(f'ground truth num: {len(gt_data)}')
        # print(f'gt dict: {gt_data}')
        # output_data = match_reordered_questions(gt_data, args.input_file)
        output_data = match_modified_questions(gt_data, args.input_file)
        

    with open(args.output_file, 'w') as fout:
        for orig_q in output_data:
            new_questions = list(output_data[orig_q])
            for new_q in new_questions:
                fout.write(orig_q + '\t' +new_q+'\n')