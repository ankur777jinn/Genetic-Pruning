# Genetic-Pruning

First Clone into the Github Repository from your notebook

    !git clone https://github.com/ankur777jinn/Genetic-Pruning.git


For Generating Importance Matrix, 

    %cd Genetic-Pruning/LLM-Pruner

Also install the requirements 

    !pip install -r requirement.txt

Enter your huggingface Token if needed via,

    from huggingface_hub import login
    login("YOUR_HUGGINGFACE_TOKEN_HERE")

Now, for generating Importance Matrix, we used TinyLlama/TinyLlama-1.1B-Chat-v1.0 as our model 

    !python hf_prune.py \
    --base_model <model_name> \
    --save_ckpt_log_name <log_name> \
    --pruning_ratio 0.25 \
    --block_wise \
    --block_mlp_layer_start 4 \
    --block_mlp_layer_end 20 \
    --block_attention_layer_start 4 \
    --block_attention_layer_end 20 \
    --pruner_type taylor \
    --taylor param_first \
    --device cuda \
    --num_examples 5 \
    --iterative_steps 1 \
    --global_pruning \
    --save_weights_importance \
    --seed 42

Importance Matrix will be saved in .pkl format. It would have an object type of dictionary. To view the contents, use:

    import pickle
    path = "<model_name>"
    print(data.keys())     # This would give you the distionary keys

To see contents against a key,

    key = "model.layers.0.self_attn.q_proj.weight"  # example
    imp_matrix = data[key]
    
    print(type(imp_matrix))
    print(getattr(imp_matrix, "shape", None))  # works if numpy/tensor
    print(imp_matrix[:5])  # show first 5 rows/elements
        

Finally, to see the overall Stats

    imp_matrix_cpu = imp_matrix.detach().cpu().numpy()      # Move your tensor to CPU + convert to numpy
    
    print("Min:", imp_matrix_cpu.min())
    print("Max:", imp_matrix_cpu.max())
    print("Mean:", imp_matrix_cpu.mean())
    print("Std:", imp_matrix_cpu.std())
    print("count_0",imp_matrix_cpu.count(0) )

After Generating Importance Matrix, for applying Genetic Algorithm to the Model using the matrix scores,

    !python pruning_ga.py \
    --model <model_name> \
    --generations 2 \
    --population 2 \
    --pruning-rate 0.2 \
    --block-size 8 \
    --mutation-rate 0.15 \
    --eval-samples 100 \
    --importance-dict <path to importance scores.pkl> \
    --importance-lambda 0.01 \
    --hf-repo <hf-username>/tinyllama-pruned-20percent \
    --output-dir ./results


