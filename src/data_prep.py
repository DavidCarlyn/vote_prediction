#####################################################
# Written by: David Carlyn & Dan Weber
# David Carlyn - Textual data and combination
# Dan Weber - Audio data
#####################################################

import os
import sys
import csv
import json
import gc
import pickle
import parselmouth
import nltk
import numpy as np
nltk.download('punkt')

from pathlib import Path
from argparse import ArgumentParser
from torchtext.vocab import GloVe
from progress.bar import Bar

TEXT_FEATURES = 200

# Assert required parameters
def run_assertions(args):
    assert os.path.isdir(args.audio_data_dir), f"{args.audio_data_dir} does not exist."
    assert os.path.isdir(args.text_data_dir), f"{args.text_data_dir} does not exist."

# Str to float middle function
# --undefined-- is set to a numerical value
def str_to_float(str_val, path):
    if str_val == '--undefined--':
        #print(f'{path} chucked undefined!')
        return -1.0
    else:
        return float(str_val)

#################################
# Written By: David Carlyn
#################################
# Extract textual features from a file
#####################################################
# Input: 
#     f: file to extract textual features
#     glove: GloVe model to create features
# Output:
#     list of sentence features
#####################################################
def process_text_file(f, glove):
    rows = []
    # Open the CSV file
    with open(f) as csv_file:
        reader = csv.reader(csv_file)
        i = -1
        row = 0
        # Iterative every row
        while row is not None:
            try:
                i += 1
                # Reading row
                row = next(reader, None)

                # Determine if row is useful
                if i == 0: continue
                if row == None:
                    #print(f"Stopped reading {f} at sentence: {i}")
                    continue
                try:
                    # Extract words
                    sentence = row[-1]
                    # Tokenize words
                    tokens = nltk.tokenize.word_tokenize(sentence)

                    # Pass words into GLOVE
                    embeddings = None
                    for k, t in enumerate(tokens):
                        if k == 0:
                            embeddings = glove[t].cpu().numpy().reshape(1, -1)
                        else:
                            new_embeddings = glove[t].cpu().numpy().reshape(1, -1)
                            embeddings = np.concatenate((embeddings, new_embeddings), axis=0)
                    row.append(embeddings.mean(0).tolist())

                # Handle exceptions
                except Exception as e:
                    print(f"Sentence: {i}, Error: {e}")
                    row = [None]
                finally:
                    if i != 0:
                        row.insert(0, i)
                        rows.append(row)
            except Exception as e:
                print(f"Sentence: {i}, Error: {e}")
                if i != 0:
                    rows.append([i, None])

    return rows

#################################
# Written By: David Carlyn
#################################
# Extract text features given directory for textual data
# Header rows are skipped (so starting rows will have # = 2)
# If a row can't be read, only the row # and a None value will be in place there [row_idx, None]
# If a row can be read, but the sentence/word can't be tokenized, the last value will be None (for no features)
# If a row can be read, and the sentence is tokenizable, then the row will be as follows:
# [row_idx, speaker_ID, speaker_role, start, stop, byte_start, byte_stop, [[word_embeddings], [word_embeddings], ...words..., [word_embeddings]]]
def extract_text_features(text_data_dir, output_data_dir):
    num_files = 0
    # Download GLOVE features
    glove = GloVe(name='twitter.27B', dim=TEXT_FEATURES)
    # Iterate through files
    for root, dirs, files in os.walk(text_data_dir):
        for f in Bar(f"Reading files from {root}").iter(files):
            num_files += 1
            # Get Case number
            case = f.split('_')[1]
            case = case.split('.')[0]

            # Process file
            rows = process_text_file(os.path.join(root, f), glove)

            # Save file
            with open(os.path.join(output_data_dir, f"text_{case}.pkl"), 'wb') as outfile:
                pickle.dump(rows, outfile)

#################################
# Written By: Dan Weber
#################################
# Extract audio features given directory for audio data.
def extract_audio_features(audio_data_dir, output_data_dir):
    audio_map = {}
    for root, dirs, files in os.walk(audio_data_dir):
        for f in files:
            year, case, sentence = f.split('_')
            sentence, ext = sentence.split('.')
            if year not in audio_map:
                audio_map[year] = {}
            if case not in audio_map[year]:
                audio_map[year][case] = []
            audio_map[year][case].append([int(sentence)+1, os.path.join(root, f)])

    cases = []
    broken_trans_files = []
    more_audio = []
    missing_files = []
    undefined_audio_files = []
    num_files = 0
    for year in audio_map:
        for case in audio_map[year]:
            num_files += 1
            sentence_features = []
            sentences = audio_map[year][case]
            sentences.sort(key=lambda x: x[0])
            for sen in Bar(f"Processing Year: {year} Case: {case}").iter(sentences):
                path = sen[1]
                
                # sound, pitch, and pulses are the 3 objects needed for Praat software.
                # The try block is because some of the files are missing.
                # The size check is to dodge corrupt files that were breaking even the try block.
                try:
                    fileSize = Path(path).stat().st_size
                    if fileSize > 2000:
                        sound = parselmouth.Sound(path)
                    else:
                        print(f'Unable to process {path}!')
                        undefined_audio_files.append(path)
                        sentence_features.append([sen[0], None])
                        continue
                except:
                    print(f'{path} is missing!')
                    missing_files.append(path)
                    sentence_features.append([sen[0], None])
                    continue

                pitch = sound.to_pitch()
                pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

                # main call to Praat through Parselmouth.  Values are all Praat defaults, perfectly acceptable for normal speech.
                # Output is a long string that gets parsed for all the features.
                voice_report_str = parselmouth.praat.call([sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)
                VPRsplit = voice_report_str.split()
                feature_list = []

                ### Pitch Features ###
                # Median pitch
                feature_list.append(str_to_float(VPRsplit[11], path))
                # Mean pitch
                feature_list.append(str_to_float(VPRsplit[15], path))
                # Pitch standard deviation
                feature_list.append(str_to_float(VPRsplit[19], path))
                # Minimum pitch
                feature_list.append(str_to_float(VPRsplit[23], path))
                # Maximum pitch
                feature_list.append(str_to_float(VPRsplit[27], path))

                ### Pulse Features ###
                # Number of pulses
                feature_list.append(str_to_float(VPRsplit[33], path))
                # Number of periods
                feature_list.append(str_to_float(VPRsplit[37], path))
                # mean period length
                feature_list.append(str_to_float(VPRsplit[40], path))
                # Period stdev
                feature_list.append(str_to_float(VPRsplit[46], path))

                ### Voicing Features ###
                # Fraction of locally unvoiced frames
                feature_list.append(str_to_float(VPRsplit[54].strip('%'), path) / 100.0)
                # Number of voice breaks
                feature_list.append(str_to_float(VPRsplit[62], path))
                # Degree of voice breaks
                feature_list.append(str_to_float(VPRsplit[67].strip('%'), path) / 100.0)

                ### Jitter Features ###
                # Jitter (local)
                feature_list.append(str_to_float(VPRsplit[76].strip('%'), path) / 100.0)
                # Jitter (local, absolute)
                feature_list.append(str_to_float(VPRsplit[80], path))
                # Jitter (rap)
                feature_list.append(str_to_float(VPRsplit[84].strip('%'), path) / 100.0)
                # Jitter (ppq5)
                feature_list.append(str_to_float(VPRsplit[87].strip('%'), path) / 100.0)
                # Jitter (ddp)
                feature_list.append(str_to_float(VPRsplit[90].strip('%'), path) / 100.0)

                ### Shimmer Features ###
                # Shimmer (local)
                feature_list.append(str_to_float(VPRsplit[94].strip('%'), path) / 100.0)
                # Shimmer (local, dB)
                feature_list.append(str_to_float(VPRsplit[98], path))
                # Shimmer (apq3)
                feature_list.append(str_to_float(VPRsplit[102].strip('%'), path) / 100.0)
                # Shimmer (apq5)
                feature_list.append(str_to_float(VPRsplit[105].strip('%'), path) / 100.0)
                # Shimmer (apq11)
                feature_list.append(str_to_float(VPRsplit[108].strip('%'), path) / 100.0)
                # Shimmer (dda)
                feature_list.append(str_to_float(VPRsplit[111].strip('%'), path) / 100.0)

                ### Harmonicity Features (of voiced only) ###
                # mean aurocorrelation
                feature_list.append(str_to_float(VPRsplit[120], path))
                # Mean noise-to-harmonics ratio
                feature_list.append(str_to_float(VPRsplit[124], path))
                # Mean harmonics-to-noise ratio
                feature_list.append(str_to_float(VPRsplit[128], path))

                sentence_features.append([int(sen[0]), feature_list])
            cases.append([int(case), sentence_features])

            if num_files % 50 == 0:
                print("Saving Cases")
                for case in cases:
                    with open(os.path.join(output_data_dir, f"audio_{case[0]}.pkl"), 'wb') as outfile:
                        pickle.dump(case[1], outfile)
                cases = []
    
    print("Saving Final Cases")
    for case in cases:
        with open(os.path.join(output_data_dir, f"audio_{case[0]}.pkl"), 'wb') as outfile:
            pickle.dump(case[1], outfile)

# Read Pickle file
def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

#################################
# Written By: David Carlyn
#################################
# combines features based on files saved in output_data_dir
def combine_features(output_data_dir):
    text_paths = []
    audio_paths = []
    # Iterate through files
    for root, dirs, files in os.walk(os.path.join(output_data_dir, 'tmp')):
        for f in files:
            # Get valid files only
            if os.path.getsize(os.path.join(root, f)) <= 0: 
                print(f)
                continue
            # Obtain case and data type
            f_type, case = f.split('_')
            case, ext = case.split('.')

            # Save paths
            if f_type == "text":
                text_paths.append([case, os.path.join(root, f)])
            elif f_type == "audio":
                audio_paths.append([case, os.path.join(root, f)])
            else:
                print(f"Error: {f} will not be computed.")

    # Sort by case
    text_paths.sort(key=lambda x: x[0])
    audio_paths.sort(key=lambda x: x[0])

    metadata = []
    all_features = []
    done = False
    text_i = 0
    audio_i = 0
    # Iterate through cases
    while not done:
        # Make sure not overflowing
        if text_i >= len(text_paths) and audio_i < len(audio_paths):
            metadata.append({ "case" : audio_paths[audio_i][0], "valid" : False })
            all_features.append(None)
            audio_i += 1
            continue
        elif text_i < len(text_paths) and audio_i >= len(audio_paths):
            metadata.append({ "case" : text_paths[text_i][0], "valid" : False })
            all_features.append(None)
            text_i += 1
            continue
        elif text_i >= len(text_paths) and audio_i >= len(audio_paths):
            done = True
            continue

        # Reading in features
        case_audio_features =  read_pkl(audio_paths[audio_i][1])
        case_text_features = read_pkl(text_paths[text_i][1])
        
        # Recording case number
        text_case = text_paths[text_i][0]
        audio_case = audio_paths[audio_i][0]

        # Only save data if we have both the text and audio data matching (case == case)
        # Only record if they match
        if text_case != audio_case:
            # Audio Case > Text Case
            if text_case < audio_case:
                metadata.append({ "case" : text_case, "valid" : False })
                all_features.append(None)
                text_i += 1
            # Audio Case < Text Case
            else:
                metadata.append({ "case" : audio_case, "valid" : False })
                all_features.append(None)
                audio_i += 1
        else:
            sen_i = 0
            sen_j = 0
            sen_done = False
            sen_meta = []
            sen_features = []
            # Iterate through sentences
            while not sen_done:
                # Ensure we are not overflowing on our sentence iteration
                if sen_i >= len(case_text_features) and sen_j < len(case_audio_features):
                    sen_features.append(None)
                    sen_meta.append({"sentence_num" : case_audio_features[sen_j][0], "valid" : False})
                    sen_j += 1
                    continue
                elif sen_i < len(case_text_features) and sen_j >= len(case_audio_features):
                    sen_features.append(None)
                    sen_meta.append({"sentence_num" : case_text_features[sen_i][0], "valid" : False})
                    sen_i += 1
                    continue
                elif sen_i >= len(case_text_features) and sen_j >= len(case_audio_features):
                    sen_done = True
                    continue

                # Extracting text and sentence features
                text_sen = case_text_features[sen_i]
                audio_sen = case_audio_features[sen_j]

                # Only record if the text sentence and audio sentence match
                if text_sen[0]-1 != audio_sen[0]+1:
                    sen_features.append(None)
                    # Text sentence is less than audio sentence
                    if text_sen[0] < audio_sen[0]:
                        sen_meta.append({ "sentence_num" : text_sen[0], "valid" : False })
                        sen_i += 1
                    # Text sentence is greater than audio sentence
                    else:
                        sen_meta.append({ "sentence_num" : audio_sen[0], "valid" : False })
                        sen_j += 1
                else:
                    # Ensure there is data there
                    if text_sen[-1] is not None:
                        # Extract raw data per sentence
                        combined = audio_sen[-1]
                        combined.extend(text_sen[-1])
                        sen_features.append(combined)

                        # Record metadata
                        sen_meta.append({
                            "valid" : True,
                            "sentence_num" : text_sen[0],
                            "speaker_id" : text_sen[1],
                            "speaker_role" : text_sen[2],
                            "sentence" : text_sen[-2]
                        })

                    # Record as Invalid data if None
                    else:
                        sen_meta.append({ "sentence_num" : text_sen[0], "valid" : False })
                        sen_features.append(None)


                    # Increment sentence idx
                    sen_i += 1
                    sen_j += 1


            # Append and save data
            metadata.append({"case": text_case, "valid" : True, "sentences" : sen_meta})
            all_features.append(sen_features)

            # Increment case idx
            text_i += 1
            audio_i += 1

    # Save Data
    with open(os.path.join(output_data_dir, 'metadata.pkl'), 'wb') as outfile:
        pickle.dump(metadata, outfile)

    with open(os.path.join(output_data_dir, 'features.pkl'), 'wb') as outfile:
        pickle.dump(all_features, outfile)


######################################
# Written By: David Carlyn & Dan Weber
######################################
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--audio_data_dir', type=str, default=os.path.join('..', 'data', 'audio'))
    parser.add_argument('--text_data_dir', type=str, default=os.path.join('..', 'data', 'text'))
    parser.add_argument('--output_data_dir', type=str, default=os.path.join('..', 'data', 'combined'))
    args = parser.parse_args()

    # Run assertions for program correctness
    run_assertions(args)

    # Create output dir if it does not exist
    if not os.path.isdir(args.output_data_dir):
        os.makedirs(args.output_data_dir)
        os.makedirs(os.path.join(args.output_data_dir, 'tmp'))

    # Extracting text features
    extract_text_features(args.text_data_dir, os.path.join(args.output_data_dir, 'tmp'))

    # Extracting audio features
    extract_audio_features(args.audio_data_dir, os.path.join(args.output_data_dir, 'tmp'))

    # Combine features
    combine_features(args.output_data_dir)
