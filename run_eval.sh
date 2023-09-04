TEMP=1.0

python eval.py --dataset trivia_qa \
    --res_file ../generated/trivia_qa/gen_val_1000_verb_1s_top1_${TEMP}.txt \
    --bin 20
