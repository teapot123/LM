TEMP=0.5

python eval.py --dataset trivia_qa \
    --res_file generated/trivia_qa/gen_val_1000_verb_1s_top2_${TEMP}.txt \
    --bin 10


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



