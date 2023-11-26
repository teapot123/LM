# TEMP=0.3

# python eval.py --dataset trivia_qa \
#     --res_file generated/trivia_qa/top1_lora_fixmlp_7b.txt \
#     --bin 10 --guess_num 1

# use matched questions
python eval.py --dataset trivia_qa \
    --res_file generated/trivia_qa/final_outputs/combine_original_new.txt \
    --bin 10 --guess_num 1 --question_match_file /shared/data2/jiaxinh3/Calibration/LM/generated/trivia_qa/augmented_questions/post_modified_val_questions1.txt


# DIRECTORY="generated/trivia_qa"

# # Loop through all .txt files in the directory
# for FILE in $DIRECTORY/*.txt; do
#     # Extract the filename without the path and extension
#     FILENAME=$(basename -- "$FILE")
#     BASENAME="${FILENAME%.*}"

#     echo ${FILENAME}

#     # Use the file name in your script
#     python eval.py --dataset trivia_qa \
#         --res_file "$DIRECTORY/$BASENAME.txt" \
#         --bin 10
# done



