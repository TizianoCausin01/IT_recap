# ConvRNN-to-ViT Dynamic RSA First Pass

## Question
Use Nayebi et al. 2022 ConvRNN activity as an in silico dynamic IT signal and test whether static ViT-L layers align with different ConvRNN timesteps, analogous to the layer-by-time RSA effect reported for monkey aIT by Xiao et al. 2025.

## Data
- Stimuli: `/Users/tizianocausin/livingstone_lab_local/Stimuli/talia_20each_tizi`
- Existing ViT-L features: `/Users/tizianocausin/metrics_II_local/models/talia_20each_tizi_vit_l_16_384_blocks.*.mlp.fc2_features_meanpool.npz`
- Output location follows the `IT_recap` convention through `config.yaml`: `/Users/tizianocausin/IT_recap_local`

## First model
Start with `rgc_intermediate`, using its IT-like layers `conv9` and `conv10`. The ConvRNN README says intermediate layers `conv9` and `conv10` are closest to cIT/aIT, and the model timestep is roughly 10 ms.

## Dimensionality
Use `--pooling mean` first. This averages spatial dimensions and makes each layer-time a channels-by-images matrix. If this is still too large or noisy, use `--pooling all --srp_dim 1000` to flatten then apply deterministic signed random projection.

## Commands
The ConvRNN repo requires TensorFlow 1.x with `tf.contrib`, so use a separate Python 3.7-era environment for extraction.

```bash
bash bash_scripts/download_convrnn_checkpoint.sh rgc_intermediate
```

```bash
MY_ENV=tiziano_mac_mini python python_scripts/scripts/extract_convrnn_features.py \
    --model_name rgc_intermediate \
    --layers conv9,conv10 \
    --folder_name talia_20each_tizi \
    --pooling mean \
    --batch_size 16 \
    --image_pres neural
```

Then run the static dRSA comparison in the normal scientific Python environment:

```bash
MY_ENV=tiziano_mac_mini python python_scripts/scripts/run_static_dRSA_convrnn_it.py \
    --target_model_name rgc_intermediate \
    --target_layer conv10 \
    --signal_RDM_metric correlation \
    --model_RDM_metric correlation \
    --RSA_metric spearman
```

The output `drsa` matrix has shape `ConvRNN time x ViT layer`. A hierarchy-like effect would appear as a systematic shift in `best_layer_name` over ConvRNN time.

For centered cosine distances, use:

```bash
MY_ENV=tiziano_mac_mini python python_scripts/scripts/run_static_dRSA_convrnn_it.py \
    --target_model_name rgc_intermediate \
    --target_layer conv10 \
    --signal_RDM_metric cosine_cnt \
    --model_RDM_metric cosine_cnt \
    --RSA_metric spearman
```

The expected result filename is `static_dRSA_cosine_cnt-cosine_cnt_talia_20each_tizi_rgc_intermediate_conv10_target_vs_vit_l_16_spearman.npz`.
