scheduler="linear constant_with_warmup constant"
lr=5e-7
model="mistralai/Mistral-7B-Instruct-v0.3" 
for sch in $scheduler
    do
    for temp in 0.05 0.1 0.15 0.2
	do
	echo "$sch $lr $model $temp"
	python3 finetuning.py --model_id $model --decay 0.01 --lr 5e-7 --temperature $temp --num_epochs 10 --lr_scheduler_type $sch --model_num_layers 16
        done

   done


