import numpy as np
import torch
import torchaudio
import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity
import scipy.io.wavfile as wav
from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Add, Input, GlobalAveragePooling1D, Dense, Activation
from keras.metrics import binary_accuracy
import Levenshtein
from spellchecker import SpellChecker
import datetime
from tensorflow.keras.optimizers import Adam, SGD
from keras import Model
from matplotlib import pyplot as plt
import keras.backend as K
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import random
import os
import time
from collections import Counter
import csv


def count_words(transcriptions):
    word_counter = Counter()
    for sentence in transcriptions:
        words = sentence.lower().split()
        word_counter.update(words)
    return word_counter

def count_unique_transcriptions(transcriptions):
    return Counter(transcriptions)

def save_word_counts_csv(word_counts, filename):
    with open(filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Count"])
        for word, count in word_counts.most_common():
            writer.writerow([word, count])


if not tf.executing_eagerly():
    tf.config.experimental_run_functions_eagerly(True)

class SpeechGAN_Wav2Vec:
    def __init__(self, clean_wav, target_transcription, alpha=1.0, epsilon=0.5, vocab_size=27, embedding_dim=256):
        self.first_attack_end_time = 0
        self.final_end_time = 0
        self.first_attack_successful = False
        self.attack_successful = False
        self.first_SNR = 0
        self.SNR = 0
        self.first_attack_transcription = None
        self.final_attack_transcription = None
        self.first_attack_epoch = 0
        self.final_attack_epoch = 0
        self.alpha = alpha
        self.epsilon = epsilon
        self.clean_wav = clean_wav
        self.target_transcription = target_transcription

        self.spell = SpellChecker()

        # Load Wav2Vec 2.0 model
        model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
        self.model_x = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        clean_audio, sampling_rate = torchaudio.load(clean_wav)
        print("TYPE OF AUDIO: ", type(clean_audio))
        print("SHAPE OF AUDIO: ", clean_audio.shape)


        # Convert target transcription to integer sequence
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9,
                      "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19,
                      "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29,
                      "Q": 30, "Z": 31}

        # Convert target transcription to integer sequence for embedding
        self.target_sequence = [self.vocab[c] for c in target_transcription if c in self.vocab]
        self.target_padded = np.pad(self.target_sequence, (0, 50 - len(self.target_sequence)), mode='constant')
        self.target_padded = np.array(self.target_padded)


        # Optimizers
        self.optimizer_g = Adam(0.0001)
        self.optimizer_d = SGD(0.01, momentum=0.9)
        # Build Generator and Discriminator
        input_shape = (clean_audio.shape[1], 1)
        # print("INPUTSHAPE: ", input_shape)
        inputs = Input(shape=input_shape)
        generator = self.build_generator(inputs)
        self.G = Model(inputs, generator, name='Generator')
        # self.G.summary()

        discriminator = self.build_discriminator(self.G(inputs))
        self.D = Model(inputs, discriminator, name='Discriminator')

        self.D.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1),
            optimizer=self.optimizer_d,
            metrics=[self.custom_acc]
        )
        # self.D.summary()

        self.gan = Model(inputs=inputs,
                         outputs=[self.G(inputs), self.D(inputs), self.G(inputs)])

        self.gan.compile(
            loss=[
                self.hinge_loss_generator,
                tf.keras.losses.binary_crossentropy,
                self.ctc_loss
            ],
            loss_weights=[0.01, 0.5, 1.0],
            optimizer=self.optimizer_g,
            run_eagerly=True
        )
        # self.gan.summary()

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_pred), K.round(y_true))

    def preprocess_audio(self, audio):
        audio_flatten = tf.reshape(audio, [-1])
        audio_clipped = tf.clip_by_value(audio_flatten, -1.0, 1.0) * 32767
        return audio_clipped.numpy().astype(np.int16)

    def transcribe_audio(self, audio):
        with torch.no_grad():
            logits = self.model_x(audio).logits
        # Decode Transcription and Probabilities
        predicted_ids = torch.argmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        # print("Transcription:", transcription)
        return logits, predicted_ids, probs, transcription

    def ctc_loss(self,  target_text, adversarial_audio):
        # print("Target text: ", target_text)
        # adversarial_audio = self.build_generator(adversarial_audio)
        target_text = self.target_transcription
        # print("Adversarial audio shape: ", adversarial_audio.shape, adversarial_audio)
        adversarial_audio = torch.tensor(adversarial_audio.numpy().squeeze(), dtype=torch.float32)

        # convert numpy array to PyTorch tensor:
        audio_tensor = torch.tensor(adversarial_audio, dtype=torch.float32)
        audio_tensor = audio_tensor.unsqueeze((0))
        #Process audio input
        inputs = self.processor(audio_tensor, sampling_rate=16000, return_tensors='pt', padding=True)
        input_values = inputs.input_values.view(1, 1, -1).squeeze(0)
        # print(f"input_values shape: {input_values.shape}")  # Should be (batch, time)
        # Get logits from the Wav2Vec2 model
        with torch.no_grad():
            logits = self.model_x(input_values).logits  # Shape: (batch, time, vocab_size)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Convert target text to token IDs
        target_encoded = self.processor.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True).input_ids

        # Define sequence lengths
        input_lengths = torch.full(size=(log_probs.shape[0],), fill_value=log_probs.shape[1], dtype=torch.long)
        target_lengths = torch.full(size=(target_encoded.shape[0],), fill_value=target_encoded.shape[1], dtype=torch.long)

        # Compute CTC Loss
        ctc_loss_fn = torch.nn.CTCLoss(blank=self.processor.tokenizer.pad_token_id, reduction='mean')
        loss = ctc_loss_fn(log_probs.permute(1, 0, 2), target_encoded, input_lengths, target_lengths)
        # print(loss.item())
        return loss.item()  # Lower loss means a better adversarial attack

    def cosine_similarity_loss(self, y_targ, y_pred):
        # audio = clean_audio.reshape(-1, 1)
        # audio = np.expand_dims(audio, axis=0)
        y_pred = self.G(clean_audio)

        _, _, _, y_pred_transcription = self.transcribe_audio(y_pred)

        y_pred_sequence = [self.vocab.get(c, 26) for c in y_pred_transcription]
        y_pred_padded = np.pad(y_pred_sequence, (0, 50 - len(y_pred_sequence)), mode='constant')

        y_pred_padded = tf.convert_to_tensor(y_pred_padded, dtype=tf.float32)

        y_targ = tf.cast(y_targ, tf.float32)
        y_targ = tf.nn.l2_normalize(y_targ, axis=-1)
        y_pred_padded = tf.nn.l2_normalize(y_pred_padded, axis=-1)

        cosine_similarity = tf.reduce_sum(y_targ * y_pred_padded, axis=-1)
        loss = 1 - cosine_similarity

        return loss * 2

    def mse_loss(self, y_targ, y_pred):
        pred_ids,y_pred_transcription = tf.numpy_function(self.get_pred_ids_flattened_fn, [y_pred], tf.float32)
        pred_ids = tf.reshape(pred_ids, [-1])
        pred_ids = tf.cast(pred_ids, tf.float32)

        # Flatten and cast target IDs
        y_targ_flat = tf.reshape(tf.cast(y_targ, tf.float32), [-1])

        # Compute Mean Squared Error
        print("pred_ids.shape",pred_ids.shape)
        print("y_targ_flat.shape",y_targ_flat.shape)
        loss = tf.reduce_mean(tf.square(pred_ids - y_targ_flat))

        return loss

    def hinge_loss_generator(self, y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.1, 0), axis=-1)

    def get_pred_ids_flattened_fn(self, adv_audio_np):
        adv_audio = torch.tensor(adv_audio_np.squeeze(), dtype=torch.float32)
        adv_audio = self.processor(adv_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        adv_audio = adv_audio.reshape(1, -1)
        logits, y_pred_prediction_ids, probs, y_pred_transcription = self.transcribe_audio(adv_audio)
        return y_pred_prediction_ids.detach().cpu().numpy().flatten(), y_pred_transcription

    def build_generator(self, inputs):
        x = Conv1D(32, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(1, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Add()([x * 0.25, inputs])
        return x

    def build_discriminator(self, generator_output):
        x = Conv1D(16, kernel_size=3, strides=2, padding='same')(generator_output)
        x = LeakyReLU()(x)
        x = Conv1D(32, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(8, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return x

    def train_discriminator(self, x_, Gx_):
        self.D.trainable = True
        real_labels = np.ones((1, 1)) * 0.9  # Real data labels
        real_loss = self.D.train_on_batch(x_, real_labels)

        # Train Discriminator on fake data (generated by the generator)
        fake_labels = np.zeros((1, 1)) + 0.1  # Fake data labels
        fake_loss = self.D.train_on_batch(Gx_, fake_labels)

        # Total discriminator loss
        d_loss = 0.25 * np.add(real_loss, fake_loss)
        # print("D_LOSS: ", d_loss)
        return d_loss

    def train_generator(self, x_):
        self.D.trainable = False
        N = x_.shape[0]  # Get the batch size
        real_label = np.ones((1, 1))  # discriminator target for generator's output
        target_text = self.target_transcription

        target_encoded = self.processor.tokenizer(target_text, return_tensors="pt", padding=True,
                                                  truncation=True).input_ids
        target_encoded = tf.convert_to_tensor(target_encoded, dtype=tf.int32)  # Convert to TensorFlow tensor
        target_encoded_batch = tf.reshape(target_encoded, (N, -1))

        print("target_encoded_batch.shape", target_encoded_batch.shape)
        print("target_encoded_batch", target_encoded_batch)

        # print("Adversarial audio input shape: ", x_.shape)

        g_loss = self.gan.train_on_batch(x_, [x_, real_label, target_encoded_batch])

        return g_loss

    def save_audio(self, filename, audio_data, sample_rate=16000):
        wav.write(filename, sample_rate, audio_data)  # Save as 16-bit PCM

    def plot_audio(self, epoch, original_audio, adversarial_audio, noise, key, file_name, sample_rate=16000):
        # Ensure audio arrays are flattened to shape (16000,)
        # original_audio = original_audio.flatten()
        adversarial_audio = adversarial_audio.flatten()
        # noise = noise.flatten()

        # Create a time axis that corresponds to the audio length
        time = np.linspace(0, len(original_audio) / sample_rate, num=len(original_audio))
        plt.figure(figsize=(10, 4))

        # Plot the original audio
        plt.plot(time, original_audio, label="Original Audio", alpha=0.7)


        # Plot the adversarial audio
        plt.plot(time, adversarial_audio, label="Adversarial Audio", alpha=0.7)

        # Plot the noise (adversarial audio - original audio)
        plt.plot(time, noise, label="Noise", alpha=0.7)

        # Add labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Original, Adversarial Audio, and Noise")
        plt.legend(loc="upper right")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base = os.path.splitext(os.path.basename(file_name))[0]
        if key == 0:
            plt.savefig(f"HPC_{file_base}_left_first_epoch_{epoch}_time_{timestamp}.png")
        elif key == 1:
            plt.savefig(f"HPC_{file_base}_left_final_epoch_{epoch}_time_{timestamp}.png")
        # plt.show()

    def is_semantically_valid(self, transcription):
        words_in_transcription = transcription.lower().split()
        if any(len(word) <= 1 for word in words_in_transcription):
            return False
        misspelled = self.spell.unknown(words_in_transcription)
        return len(misspelled) == 0

    def computeSNR(self,audio_0_numpy, noise):
        signal_power = np.sum(audio_0_numpy ** 2)
        noise_power = np.sum(noise ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))  # epsilon to avoid division by zero
        print(f"SNR of adversarial audio: {snr_db:.2f} dB")
        return snr_db

    def train(self, file_name, epochs=1000):
        file_base = os.path.splitext(os.path.basename(file_name))[0]
        print("file_base")
        print(file_base)
        print("target transcription", self.target_transcription)

        input_values = self.processor(clean_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        input_values = input_values.reshape(1, -1)

        _, original_prediction_ids, _, original_transcription = self.transcribe_audio(input_values)
        print("Original transcription:", original_transcription)

        # original_length = len(original_transcription)
        # threshold = original_length
        # print("threshold", threshold)

        tgt_ids = [self.vocab[c] for c in self.target_transcription]

        # print("ORINAL GRALSIFJ: ", original_transcription)
        audio = clean_audio.reshape(-1, 1)  # Reshape raw audio for model input (batch size 1)
        X_batch = np.expand_dims(audio, axis=0)  # Expand to match the batch size (1, ...)
        saved_untargeted = 0
        for epoch in range(epochs):
            print("==========================================================================================")
            print("==========================================================================================")
            print("epoch", epoch)

            Gx = self.G.predict(X_batch)

            # adv_audio =  Gx# + np.expand_dims(audio, axis=0)
            adv_audio = np.clip(Gx, -1.0, 1.0)


            print("Original_max, Original_min", max(audio), min(audio))
            # adv_audio = Gx
            print("Adv_audio_max, Adv_audio_min", max(adv_audio[0]), min(adv_audio[0]))
            d_loss = self.train_discriminator(X_batch, adv_audio)
            losses  = self.train_generator(X_batch)
            # Print individual losses
            total_loss = losses[0]
            generator_loss = losses[1]
            discriminator_loss = losses[2]
            ctc_loss = losses[3]

            ### CHECKING NOISE MAGNITUDES::

            audio_0_numpy = audio.cpu().detach().numpy()
            noise = adv_audio[0]  - audio_0_numpy
            print("noise_max, noise_min", max(noise), min(noise))

            Adv_torch = torch.tensor(adv_audio, dtype=torch.float32)
            Adv_torch = Adv_torch.squeeze(-1)

            Adv_torch = torch.tensor(Adv_torch.numpy(), dtype=torch.float32)
            Adv_torch = self.processor(Adv_torch, sampling_rate=16000, return_tensors="pt", padding=True).input_values
            Adv_torch = Adv_torch.reshape(1, -1)

            with torch.no_grad():
                logits = self.model_x(Adv_torch).logits
            # Decode Transcription and Probabilities
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            lev_distance = Levenshtein.distance(original_transcription, transcription)
            # print(f"Levenshtein distance: {lev_distance} / Threshold: {threshold}")

            print(f"EPOCH {epoch}:  Adversarial Audio: {transcription}      "
                  f"Tot_loss: {total_loss},     Generator_loss: {generator_loss},   "
                  f"Discriminator_LOSS: {discriminator_loss},  CTC_LOSS: {ctc_loss}")

            if saved_untargeted == 0 and lev_distance >= 1:
                self.first_attack_end_time = time.time()
                self.first_attack_epoch = epoch
                self.first_attack_successful = True
                self.first_attack_transcription = transcription
                saved_untargeted = 1
                print("Untargeted audio SNR:")
                self.first_SNR = self.computeSNR(audio_0_numpy, noise)
                print("Adversarial Transcription: ", transcription, "; Original Transcription: ",
                      original_transcription)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_audio(f"untargeted_{file_base}_left_first_epoch_{epoch}_{timestamp}.wav", adv_audio[0], sample_rate=16000)

                audio_0_numpy = audio.cpu().detach().numpy()
                Adv_numpy = np.array(adv_audio[0]).flatten()
                self.plot_audio(epoch, audio_0_numpy, Adv_numpy, noise,0, file_name=file_name)


            # if (epoch % 20 == 0 and epoch != 0) or epoch == epochs - 1:
            #     Adv_numpy = np.array(adv_audio[0]).flatten()
            #     self.plot_audio(epoch, audio_0_numpy, Adv_numpy, noise)

            if lev_distance >= 1 and len(transcription) > 1:
                print("Threshold exceeded — untargeted attack successful.")
                if self.is_semantically_valid(transcription):
                    self.final_end_time = time.time()
                    self.final_attack_epoch = epoch
                    self.attack_successful = True
                    self.final_attack_transcription = transcription
                    self.SNR = self.computeSNR(audio_0_numpy, noise)
                    print("Adversarial Transcription: ", transcription, "; Original Transcription: ",
                          original_transcription)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save_audio(f"untargeted_{file_base}_left_final_epoch_{epoch}_{timestamp}.wav", adv_audio[0], sample_rate=16000)

                    audio_0_numpy = audio.cpu().detach().numpy()
                    Adv_numpy = np.array(adv_audio[0]).flatten()
                    self.plot_audio(epoch, audio_0_numpy, Adv_numpy, noise, 1, file_name=file_name)
                    break
                else:
                    print("Semantically invalid transcription — skipping.")
                    print("Adversarial Transcription: ", transcription, "; Original Transcription: ",
                          original_transcription)


folder_path = "augmented_dataset/left"
target_transcription = ""  # Leave blank for untargeted attack
num_runs = 100

# Get a list of all .wav files in the folder
all_audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
# random.shuffle(all_audio_files)
selected_audio_files = all_audio_files[:num_runs]

first_success_count = 0
success_count = 0
total_first_attack_time = 0
total_attack_time = 0
total_first_SNR = 0
total_SNR = 0
first_transcriptions = []
final_transcriptions = []
first_attack_epochs = []
final_attack_epochs = []


for idx, file_name in enumerate(selected_audio_files):
    start_time = time.time()

    clean_wav_path = os.path.join(folder_path, file_name)
    print(f"\n\n--- Running attack {idx + 1}/{num_runs} on file: {file_name} ---")

    # Load clean audio
    clean_audio, sampling_rate = torchaudio.load(clean_wav_path)

    speech_gan = SpeechGAN_Wav2Vec(
        clean_wav=clean_wav_path,
        target_transcription=target_transcription
    )
    speech_gan.train(file_name, epochs=1000)

    first_attack_end_time = speech_gan.first_attack_end_time
    final_end_time = speech_gan.final_end_time
    run_first_attack_duration = first_attack_end_time - start_time
    run_duration = final_end_time - start_time
    total_first_attack_time += run_first_attack_duration
    if speech_gan.attack_successful:
        total_attack_time += run_duration
    first_attack_epoch = getattr(speech_gan, "first_attack_epoch", None)
    final_attack_epoch = getattr(speech_gan, "final_attack_epoch", None)
    first_attack_epochs.append(first_attack_epoch)
    final_attack_epochs.append(final_attack_epoch)


    total_first_SNR += speech_gan.first_SNR
    total_SNR += speech_gan.SNR

    print(f"Run time for first attack {idx + 1}: {run_first_attack_duration:.2f} seconds")
    print(f"Run time for attack {idx + 1}: {run_duration:.2f} seconds")

    if speech_gan.first_attack_successful:
        first_transcriptions.append(speech_gan.first_attack_transcription)
        first_success_count += 1
        print(f"First Attack {idx + 1} SUCCESSFUL")
    else:
        print(f"First Attack {idx + 1} FAILED")

    if speech_gan.attack_successful:
        final_transcriptions.append(speech_gan.final_attack_transcription)
        success_count += 1
        print(f"Attack {idx + 1} SUCCESSFUL")
    else:
        print(f"Attack {idx + 1} FAILED")


first_success_rate = (first_success_count / num_runs) * 100
success_rate = (success_count / num_runs) * 100
average_first_time = total_first_attack_time / num_runs
average_time = total_attack_time / success_count
average_first_SNR = total_first_SNR / num_runs
average_SNR = total_SNR / success_count
first_word_counts = count_unique_transcriptions(first_transcriptions)
final_word_counts = count_unique_transcriptions(final_transcriptions)
valid_first_epochs = [e for e in first_attack_epochs if e is not None]
valid_final_epochs = [e for e in final_attack_epochs if e is not None]
avg_first_epoch = sum(valid_first_epochs) / len(valid_first_epochs) if valid_first_epochs else 0
avg_final_epoch = sum(valid_final_epochs) / success_count if valid_final_epochs else 0

print(f"\n\nFirst Success rate over {num_runs} samples: {first_success_rate:.2f}%")
print(f"Success rate over {num_runs} samples: {success_rate:.2f}%")
print(f"Average runtime per attack: {average_first_time:.2f} seconds")
print(f"Average runtime per attack: {average_time:.2f} seconds")
print(f"Average SNR of first attack adversarial audio: {average_first_SNR:.2f} dB")
print(f"Average SNR of adversarial audio: {average_SNR:.2f} dB")
print(f"\nAverage Epoch of First Attack Success: {avg_first_epoch:.2f}")
print(f"Average Epoch of Final Attack Success: {avg_final_epoch:.2f}")

print("\n\n--- Word Frequency in FIRST Attack Transcriptions ---")
for word, count in first_word_counts.most_common():
    print(f"{word}: {count}")
print("\n\n--- Word Frequency in FINAL Attack Transcriptions ---")
for word, count in final_word_counts.most_common():
    print(f"{word}: {count}")

with open("LEFT_attack_transcriptions.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "File Name", "First Attack Transcription", "Final Attack Transcription",
        "First Attack Epoch", "Final Attack Epoch"
    ])

    for idx, file_name in enumerate(selected_audio_files):
        first = first_transcriptions[idx] if idx < len(first_transcriptions) else ""
        final = final_transcriptions[idx] if idx < len(final_transcriptions) else ""
        first_epoch = first_attack_epochs[idx] if idx < len(first_attack_epochs) else ""
        final_epoch = final_attack_epochs[idx] if idx < len(final_attack_epochs) else ""
        writer.writerow([file_name, first, final, first_epoch, final_epoch])

    # Empty row for separation
    writer.writerow([])

    # Average statistics row
    writer.writerow(["Average Stats"])
    writer.writerow(["First Success Rate (%)", f"{first_success_rate:.2f}"])
    writer.writerow(["Final Success Rate (%)", f"{success_rate:.2f}"])
    writer.writerow(["Average First Attack Time (s)", f"{average_first_time:.2f}"])
    writer.writerow(["Average Final Attack Time (s) for successful attacks", f"{average_time:.2f}"])
    writer.writerow(["Average First SNR (dB)", f"{average_first_SNR:.2f}"])
    writer.writerow(["Average Final SNR (dB) for successful attacks", f"{average_SNR:.2f}"])
    writer.writerow(["Average First Attack Epoch", f"{avg_first_epoch:.2f}"])
    writer.writerow(["Average Final Attack Epoch for successful attacks", f"{avg_final_epoch:.2f}"])

save_word_counts_csv(first_word_counts, "LEFT_first_attack_word_counts.csv")
save_word_counts_csv(final_word_counts, "LEFT_final_attack_word_counts.csv")