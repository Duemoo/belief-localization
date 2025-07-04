# When this run completes, compare individual case results
import json

# Load a single case from each experiment
with open('/mnt/sda/hoyeon/belief-localization/results/gpt2-medium/ROME_outputs_cf_layer-[12]_fact_-subject_last_v_lr-0.5_kl_fa-1.0/case_0.json', 'r') as f:
    rome_case = json.load(f)
    
with open('/mnt/sda/hoyeon/belief-localization/results/gpt2-medium/ROME_NOISE_outputs_cf_layer-[12]_kl_fa-1.0/case_0.json', 'r') as f:
    noise_case = json.load(f)

# Check if the model predictions actually differ
print("ROME post-edit rewrite prob:", rome_case['post']['rewrite_prompts_probs'][0])
print("NOISE post-edit rewrite prob:", noise_case['post']['rewrite_prompts_probs'][0])