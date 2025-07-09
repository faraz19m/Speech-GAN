# Speech-GAN: Black-Box Attack against Automatic Speech Recognition Systems

This repository contains the code for the paper **"Speech-GAN: Black-Box Attack against Automatic Speech Recognition Systems."**

## Repository Structure

- `Clean-Audios/` – Contains folders for each of the 10 commands used in the paper. Each folder contains audios stating the respective command.
- `Adversarial-Audios/` – Contains folders for the two kinds of untargeted adversarial attacks performed on the clean audios.
- `Adversarial-Audios/Basic-Adversarial-Audios/` – Contains successful basic adversarial attack audios.
- `Adversarial-Audios/Semantically-Constrained-Adversarial-Audios/` – Contains successful semantically constrained adversarial attack audios.
- `Speech-GAN.py` – Python source code file for the Speech-GAN attack.
- `Waveforms/` - Contains the graphs of waveforms for the basic and semantically constrained attack audios.
- `csv/` - Contains CSV files giving the words and their frequencies post successful attacks.

## Reproducibility Notice

Please note that **Speech-GAN includes inherent randomness**. Because of this, running the code multiple times may lead to **slightly different results** from those reported in the paper.

To improve reproducibility, consider setting fixed random seeds for libraries like NumPy, TensorFlow, and PyTorch. We also recommend storing the seeds used for any particular run in your logs or experiment tracker.

For any questions or contributions, feel free to open an issue or pull request.
