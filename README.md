# VGGish flow
## General steps
### VGGish-SppechCommand-1-GenSpec
This file builds a dataframe containing the path of all of our WAV files.  For each file, `vggish_input.wavfile_to_examples` is used to convert the WAV to a mel spectogram.  Each spectogram is a `96 x 64` matrix, and these are stored as a `n x 96 x 64` numpy matrix.  Not all WAV files are long enough to be processed by this function, so invalid conversions are flagged in the dataframe.  The associated cells in the numpy matrix are left as 0 value cells.  The dataframe is dumped to `wavfile_df.csv` and the spectograms are dumped to `wavfile_spec.dat`.  

The shape information is not saved by numpy, so to load the spectogram data you'll need to reshape it:
```
with open('wavfile_spec.dat', 'rb') as f:
    audio_data = np.fromfile(f)
audio_array = audio_data.reshape((-1, 96, 64))
```

## Embedding method
In this approach, VGGish is used to transform each mel spectogram into a `1 x 128` embedding vector.  This vector contains the features extracted by the VGGish network and can be used in place of the full mel spectogram.  We then build a XGBoost classifier that uses these embeddings to classify the audio.

### VGGish-SpeechCommand-Embedding-2-GenEmbed
This file loads the mel spectograms and runs each spectogram through the VGGish network in order to generate the embedding vector.  The audio data is processed in chunks due to memory requirements.  The embedding is added to the dataframe and saved as `wavfile_embed.csv`.

### VGGish-SpeechCommand-Embedding-3-Model
This file loads the dataframe, uses a `LabelEncoder` to encode the categorical labels, and then trains a `XGBClassifier` to classify the embedding vectors as their original labels.

## Transfer learning
### VGGish-SpeechCommand-TransferLearn-2
Using the spectrograms generated previously, this adds a trainable layer onto a frozen copy of VGGish and then trains and evaluates the model.

## Warm start
### VGGish-SpeechCommand-WarmStart-2
Using the spectrograms generated earlier, this extends the VGGish network by adding a layer, then re-trains the entire network, and then evaluates the model.

