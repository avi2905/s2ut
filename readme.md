# Speech To Speech Unit translation
Implementation for speech-to-unit translation (S2UT) Direct speech-to-speech translation with discrete units 
### Steps:
1. Prepare two folders, $SRC_AUDIO and $TGT_AUDIO, with ${SPLIT}/${SAMPLE_ID}.wav for source and target speech under each folder, separately. Note that for S2UT experiments, target audio sampling rate should be in 16,000 Hz, and for S2SPECT experiments, target audio sampling rate is recommended to be in 22,050 Hz.
2. To prepare target discrete units for S2UT model training, see [Link](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit) for pre-trained k-means models, checkpoints, and instructions on how to decode units from speech. Set the output target unit files (--out_quantized_file_path) as ${TGT_AUDIO}/${SPLIT}.txt. In Lee et al. 2021, we use 100 units from the sixth layer (--layer 6) of the HuBERT Base model.
3. For Data Preparartion for training s2ut model
   - Set --reduce-unit for training S2UT reduced model
   - Pre-trained vocoder and config ($VOCODER_CKPT, $VOCODER_CFG) can be downloaded from the Pretrained Models section. They are not required if --eval-inference is not going to be set during model training.
4.  For training s2ut model
    - Adjust --update-freq accordingly for different #GPUs. In the above we set --update-freq 4 to simulate training with 4 GPUs.
    - Set --n-frames-per-step 5 to train an S2UT stacked system with reduction ratio r=5. (Use $DATA_ROOT prepared without --reduce-unit.)
    - (optional) one can turn on tracking MCD loss during training for checkpoint selection by setting --eval-inference --eval-args '{"beam": 1, "max_len_a": 1}' --best-checkpoint-metric mcd_loss. It is recommended to sample a smaller subset as the validation set as MCD loss computation is time-consuming.
5. For Inference Generate unit sequences on test Data & Convert unit sequences to waveform.
 
  ## Code Required:
*1. Required Packages and Library Installation:*

bashpip install fairseq

git clone https://github.com/pytorch/fairseq

pip install sentencepiece


*2. Setup fairseq:*

For Colab Environment

bash
cd /content/fairseq

pip install --editable /content/fairseq

python setup.py build develop


*3. File Downloads*

Includes 
1. Fleurs dataset
2. Pretrained hu-bert base for english
3. K-Mean Quantized model for english
4. Vocoder for speech resynthesis & its config 

bash
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt

wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin

unzip /content/eng/dev.zip -d /content/eng

wget -P /content/voco https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000

wget -P /content/voco https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json

*4. Manifest Files Script:*

Creates Manifest file based the dataset for Data Formatting step
python
manifest_path="/manifest.txt"

def get_audio_files(manifest_path: str):
    fnames, sizes = [], []
    with open(manifest_path, "r") as f:
        root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert (
                len(items) == 2
            ), f"File must have two columns separated by tab. Got {line}"
            fnames.append(items[0])
            sizes.append(int(items[1]))
    return root_dir, fnames, sizes

get_audio_files(manifest_path)

*5. Quantization with k-means using pretrained Hu-bert ENG and k-mean-100 model:*

Creates units from target speech data to train s2ut model
bash
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py
--feature_type hubert --kmeans_model_path /km.bin
--acoustic_model_path /hubert_base_ls960.pt
--layer 6 --manifest_path $MANIFEST
--out_quantized_file_path $OUT_QUANTIZED_FILE --extension ".wav"

*6. Formatting Speech Data for Model Training:*

bash
python /fairseq/examples/speech_to_speech/preprocessing/prep_s2ut_data.py
--source /content/hind --target-dir /content/eng
--data-split dev --output-root /content/audio
--reduce-unit --vocoder-checkpoint /g_00500000 --vocoder-cfg /config.json






*7. S2UT Model Training:*

bash
fairseqain /content/req --config-yaml config.yaml
--task speech_to_speech
--target-is-code
--target-code-size 100
--vocoder code_hifigan
--criterion speech_to_unitlabel-smoothing 0.2
--arch s2ut_transformer_fisher
--share-decoder-input-output-embed
--dropout 0.1 --attentionout 0.1
--relu-dropout 0.1
--train-subset train
---subset dev
--save-dir /content/out
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7
--warmup-updates 10000 --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0
--max-update 350 --max-tokens 4000 --max-target- 3000 --update-freq 1
--seed 1 --fp16 --num-workers 8 --disable-validation


*8. Inference:*

bash
fairseq-generate /content/reqconfig-yaml config.yaml
--task speech_to_speech
--target-is-code
--target-code-size 100
--vocoder code_hifigan
--path /content/out/checkpoint_last.pt
--gen-subset train
--max-tokens 4000 --beam 10 --max-len-a 1
--results-path /content/req


*9. Unit to Waveform Conversion:*

bash
grep "^D\-" /content/sample_data/generate-train.txt | sed 's/^D-//ig' | sort -nk1 | cut -f3 > /content/sample_data/generate-train.unit

python /content/fairseq/examples/speech_to_speech/generate_waveform_from_code.py
--in-code-file /content/sample_data/generate-train.unit
--vocoder /content/voco/g_00500000
--vocoder-cfg /content/voco/config.json
--results-path /content/voco/out --dur-prediction

## Additional Resource
Script to create manifest file for the audio dataset
python
import os
import soundfile as sf

root_directory = r"C:\Users\aviar\Desktop\__pycache__\dev"
manifest_path = r"C:\Users\aviar\Desktop\__pycache__\manifest.txt"





# Function to create the manifest file
def create_manifest(root_directory, manifest_path):
    with open(manifest_path, "w") as f:
        # Write the root directory as the first line of the manifest file
        f.write(f"{root_directory}\n")

        # Walk through the root directory
        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.endswith(".wav") or file.endswith(".mp3"):
                    # Construct the relative path to the audio file
                    relative_path = os.path.relpath(os.path.join(root, file), root_directory)

                    # Get the number of frames in the audio file
                    audio_path = os.path.join(root, file)
                    try:
                        with sf.SoundFile(audio_path, "r") as audio_file:
                            num_frames = len(audio_file)
                    except Exception as e:
                        print(f"Error reading {audio_path}: {str(e)}")
                        num_frames = 0

                    # Write the relative path and number of frames to the manifest file
                    f.write(f"{relative_path}\t{num_frames}\n")

# Set the root directory and manifest path


# Create the manifest file
create_manifest(root_directory, manifest_path)

print(f"Manifest file '{manifest_path}' created.")


Script to convert filenames of source language to same as that of target (Model Requirement)
python
import os
import pandas as pd
import shutil

# Define the paths to the 'eng' and 'hind' folders
english_folder = 'C:/Users/aviar/Desktop/fluers/eng'
hindi_folder = 'C:/Users/aviar/Desktop/fluers/hind'
eng_tsv = 'C:/Users/aviar/Desktop/fluers/eng/eng.tsv'
hind_tsv = 'C:/Users/aviar/Desktop/fluers/hind/hind.tsv'
output_folder_hind = 'C:/Users/aviar/Desktop/fluers/common_audio/hind'
output_folder_eng = 'C:/Users/aviar/Desktop/fluers/common_audio/eng'   # Define the output folder

# Load the TSV files into dataframes
eng_df = pd.read_csv(eng_tsv, sep='\t', header=None, names=['audio_id', 'audio_filename', 'context', 'c1', 'c2', 'frames', 'Gender'])
eng_df.drop_duplicates(subset=['audio_id'], keep='first', inplace=True)
eng_df = eng_df[eng_df['Gender'] == 'FEMALE']
print(eng_df.shape)

hind_df = pd.read_csv(hind_tsv, sep='\t', header=None, names=['audio_id', 'audio_filename', 'context', 'c1', 'c2', 'frames', 'Gender'])
hind_df.drop_duplicates(subset=['audio_id'], keep='first', inplace=True)
hind_df = hind_df[hind_df['Gender'] == 'FEMALE']
print(hind_df.shape)

common_audio_ids = set(eng_df['audio_id']).intersection(set(hind_df['audio_id']))




# Iterate through common audio IDs and copy the corresponding files
for audio_id in common_audio_ids:
    eng_filename = eng_df.loc[eng_df['audio_id'] == audio_id, 'audio_filename'].values[0]
    hind_filename = hind_df.loc[hind_df['audio_id'] == audio_id, 'audio_filename'].values[0]

    eng_filepath = os.path.join(english_folder, eng_filename)
    hind_filepath = os.path.join(hindi_folder, hind_filename)

    # Check if both files exist before copying
    if os.path.exists(eng_filepath) and os.path.exists(hind_filepath):
        shutil.copy(eng_filepath, os.path.join(output_folder_eng, eng_filename))
        shutil.copy(hind_filepath, os.path.join(output_folder_hind, eng_filename))

print("Common audio files have been copied to the output folder:")

Script of to convert transcript of multiple speech in a language to another and store in the same file
python
import os
import pandas as pd
from googletrans import Translator

# Define the paths to the 'hind' folder and the input TSV file
hind_folder = 'C:/Users/aviar/Desktop/fluers/hind'
hind_tsv = 'C:/Users/aviar/Desktop/fluers/hind/hind.tsv'

# Load the TSV file into a DataFrame
hind_df = pd.read_csv(hind_tsv, sep='\t', header=None, names=['audio_id', 'audio_filename', 'context', 'c1', 'c2', 'frames', 'Gender'])

# Initialize the translator
translator = Translator()

# Translate the 'context' column from Hindi to English
hind_df['context_english'] = hind_df['context'].apply(lambda text: translator.translate(text, src='hi', dest='en').text)

# Define the path for the output TSV file
output_tsv = 'C:/Users/aviar/Desktop/fluers/hind/hind_english.tsv'

# Save the DataFrame with translated text to a new TSV file
hind_df.to_csv(output_tsv, sep='\t', index=False, header=False)

print(f"Translation complete. Results saved to {output_tsv}")

Script to create Audio speeches for translated scripts using gTTs
python
import os
import pandas as pd
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal

# Define the paths and filenames
hind_tsv = 'C:/Users/aviar/Desktop/fluers/hind/hind_english.tsv'
output_folder = 'C:/Users/aviar/Desktop/fluers/hind'
output_audio_folder = os.path.join(output_folder, 'eng_audios')

# Load the TSV file into a DataFrame
hind_df = pd.read_csv(hind_tsv, sep='\t', header=None, names=['audio_id', 'audio_filename', 'context', 'c1', 'c2', 'frames', 'Gender','context_english'])

# Function to convert English text to speech and save as audio
def text_to_speech_and_save(row):
    english_text = row['context_english']
    audio_filename = row['audio_filename'].replace('.wav', '.mp3')  # Keep the .mp3 file extension for gTTS

    # Convert English text to speech and save as audio
    output_mp3_path = os.path.join(output_audio_folder, audio_filename)
    tts = gTTS(text=english_text, lang='en')
    tts.save(output_mp3_path)
    
    # Convert mp3 file to wav
    audio = AudioSegment.from_mp3(output_mp3_path)
    output_wav_path = output_mp3_path.replace('.mp3', '.wav') 
    audio.export(output_wav_path, format="wav")

    # Resample the wav file to 16000 Hz
    rate, data = wavfile.read(output_wav_path)
    resampled_data = signal.resample(data, int(16000/rate*len(data)))
    wavfile.write(output_wav_path, 16000, resampled_data.astype(np.int16))

# Ensure the output audio folder exists
os.makedirs(output_audio_folder, exist_ok=True)

# Apply the text_to_speech_and_save function to each row
hind_df.apply(text_to_speech_and_save, axis=1)
#dependency issue file not found error 
#pip install ffmpeg-downloader
#ffdl install --add-path
