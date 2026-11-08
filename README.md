# EVNet-Tiny

EVNet-Tiny is a PyTorch image-classification experiment for the 200-class
Tiny ImageNet dataset. The project is currently organized as a Google Colab
notebook, [`EVNet_Tiny.ipynb`](./EVNet%20Tiny%20v1/EVNet_Tiny.ipynb).

The notebook prepares the dataset, trains a compact hybrid ConvNeXt/attention
model, and includes an experimental fine-tuning stage using S²-FracMix,
MixUp, CutMix, and EMA (exponential moving average) weights.

## Requirements

- Google Colab with a GPU runtime recommended
- Google Drive access for checkpoints
- PyTorch and Torchvision
- NumPy and tqdm

The notebook imports `google.colab` and uses Colab paths, so it is designed to
run in Google Colab rather than as a standalone local script.

## Running the notebook

1. Open `EVNet Tiny v1/EVNet_Tiny.ipynb` in Google Colab.
2. Select a GPU runtime and grant access to Google Drive when prompted.
3. Run the notebook cells in order. The first cell downloads and extracts
   Tiny ImageNet under `/content/tiny-imagenet-200`.
4. The baseline training configuration is set to 200 epochs. Its checkpoints
   are saved under
   `/content/drive/MyDrive/tinyimagenet_v2_checkpoints/tinyhybrid/`.
5. The S²-FracMix fine-tuning stage runs for up to 30 epochs and saves
   experiment checkpoints under
   `/content/drive/MyDrive/tinyimagenet_v2_checkpoints/evnet_s2fracmix_finetune_v1/`.

The notebook's saved output shows a baseline top-1 accuracy of 67.40%. The
fine-tuning run shown in the notebook reached epoch 4/30; its best result in
that output is 67.21% top-1.

## Resuming fine-tuning

The fine-tuning cell has `RESUME_FROM_LAST = True` and restores the latest
checkpoint from `last.pth` when it is present. In Colab, ensure the same
Google Drive is mounted and run the notebook cells needed to recreate the
datasets, model, and training objects, then rerun the fine-tuning cell. It
will resume at the epoch after the saved checkpoint. If no checkpoint exists,
the cell starts a new fine-tuning run from the baseline checkpoint.

Keep the baseline and fine-tuning checkpoint directories intact when
resuming. Deleting or replacing these files prevents restoration of the
corresponding training state.
