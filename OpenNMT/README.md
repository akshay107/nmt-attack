To run the attack

python ~/abijith/OpenNMT-py_old/attack.py --model MODEL_PATH --src SRC_PATH --tgt PRED_TGT_PATH --output_adv OUTPUT_FILE --attack_type ATTACK_TYPE --position TRAVERSAL_TYPE --gpu GPU_ID

There are two attack types: fthotflip (HotFlip) and ftsoftmax (Soft-Att). TRAVERSAL_TYPE can be either 0 (for Min-Grad) or 1 (for random). 
