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

def process_text_file(f, glove):
    rows = []
    with open(f) as csv_file:
        reader = csv.reader(csv_file)
        i = -1
        row = 0
        while row is not None:
            try:
                i += 1
                row = next(reader, None)
                if i == 0: continue
                if row == None:
                    #print(f"Stopped reading {f} at sentence: {i}")
                    continue
                try:
                    sentence = row[-1]
                    tokens = nltk.tokenize.word_tokenize(sentence)
                    embeddings = None
                    for k, t in enumerate(tokens):
                        if k == 0:
                            embeddings = glove[t].cpu().numpy().reshape(1, -1)
                        else:
                            new_embeddings = glove[t].cpu().numpy().reshape(1, -1)
                            embeddings = np.concatenate((embeddings, new_embeddings), axis=0)
                    row.append(embeddings.mean(0).tolist())
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

# Extract text features given directory for textual data
# Header rows are skipped (so starting rows will have # = 2)
# If a row can't be read, only the row # and a None value will be in place there [row_idx, None]
# If a row can be read, but the sentence/word can't be tokenized, the last value will be None (for no features)
# If a row can be read, and the sentence is tokenizable, then the row will be as follows:
# [row_idx, speaker_ID, speaker_role, start, stop, byte_start, byte_stop, [[word_embeddings], [word_embeddings], ...words..., [word_embeddings]]]
def extract_text_features(text_data_dir, output_data_dir):
    num_files = 0
    glove = GloVe(name='twitter.27B', dim=TEXT_FEATURES)
    for root, dirs, files in os.walk(text_data_dir):
        for f in Bar(f"Reading files from {root}").iter(files):
            num_files += 1
            case = f.split('_')[1]
            case = case.split('.')[0]

            rows = process_text_file(os.path.join(root, f), glove)
            with open(os.path.join(output_data_dir, f"text_{case}.pkl"), 'wb') as outfile:
                pickle.dump(rows, outfile)

# Extract audio features given directory for audio data
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

                voice_report_str = parselmouth.praat.call([sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)
                VPRsplit = voice_report_str.split()
                feature_list = []

                feature_list.append(str_to_float(VPRsplit[11], path))
                feature_list.append(str_to_float(VPRsplit[15], path))
                feature_list.append(str_to_float(VPRsplit[19], path))
                feature_list.append(str_to_float(VPRsplit[23], path))
                feature_list.append(str_to_float(VPRsplit[27], path))

                feature_list.append(str_to_float(VPRsplit[33], path))
                feature_list.append(str_to_float(VPRsplit[37], path))
                feature_list.append(str_to_float(VPRsplit[40], path))
                feature_list.append(str_to_float(VPRsplit[46], path))

                feature_list.append(str_to_float(VPRsplit[54].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[62], path))
                feature_list.append(str_to_float(VPRsplit[67].strip('%'), path) / 100.0)

                feature_list.append(str_to_float(VPRsplit[76].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[80], path))
                feature_list.append(str_to_float(VPRsplit[84].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[87].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[90].strip('%'), path) / 100.0)

                feature_list.append(str_to_float(VPRsplit[94].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[98], path))
                feature_list.append(str_to_float(VPRsplit[102].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[105].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[108].strip('%'), path) / 100.0)
                feature_list.append(str_to_float(VPRsplit[111].strip('%'), path) / 100.0)

                feature_list.append(str_to_float(VPRsplit[120], path))
                feature_list.append(str_to_float(VPRsplit[124], path))
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

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def combine_features(output_data_dir):
    text_paths = []
    audio_paths = []
    for root, dirs, files in os.walk(os.path.join(output_data_dir, 'tmp')):
        for f in files:
            if os.path.getsize(os.path.join(root, f)) <= 0: 
                print(f)
                continue
            f_type, case = f.split('_')
            case, ext = case.split('.')
            if f_type == "text":
                text_paths.append([case, os.path.join(root, f)])
            elif f_type == "audio":
                audio_paths.append([case, os.path.join(root, f)])
            else:
                print(f"Error: {f} will not be computed.")
    text_paths.sort(key=lambda x: x[0])
    audio_paths.sort(key=lambda x: x[0])
    metadata = []
    all_features = []
    done = False
    text_i = 0
    audio_i = 0
    while not done:
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

        case_audio_features =  read_pkl(audio_paths[audio_i][1])
        case_text_features = read_pkl(text_paths[text_i][1])
        text_case = text_paths[text_i][0]
        audio_case = audio_paths[audio_i][0]
        if text_case != audio_case:
            if text_case < audio_case:
                metadata.append({ "case" : text_case, "valid" : False })
                all_features.append(None)
                text_i += 1
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
            while not sen_done:
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

                text_sen = case_text_features[sen_i]
                audio_sen = case_audio_features[sen_j]
                if text_sen[0]-1 != audio_sen[0]+1:
                    sen_features.append(None)
                    if text_sen[0] < audio_sen[0]:
                        sen_meta.append({ "sentence_num" : text_sen[0], "valid" : False })
                        sen_i += 1
                    else:
                        sen_meta.append({ "sentence_num" : audio_sen[0], "valid" : False })
                        sen_j += 1
                else:
                    if text_sen[-1] is not None:
                        combined = audio_sen[-1]
                        combined.extend(text_sen[-1])
                        sen_features.append(combined)
                        sen_meta.append({
                            "valid" : True,
                            "sentence_num" : text_sen[0],
                            "speaker_id" : text_sen[1],
                            "speaker_role" : text_sen[2],
                            "sentence" : text_sen[-2]
                        })
                        if text_sen[0] - audio_sen[0] != 0:
                            print("ERROR")
                    else:
                        sen_meta.append({ "sentence_num" : text_sen[0], "valid" : False })
                        sen_features.append(None)

                    if text_sen[0] != len(sen_meta):
                        print("Error")


                    sen_i += 1
                    sen_j += 1


            metadata.append({"case": text_case, "valid" : True, "sentences" : sen_meta})
            all_features.append(sen_features)

            text_i += 1
            audio_i += 1

    with open(os.path.join(output_data_dir, 'metadata.pkl'), 'wb') as outfile:
        pickle.dump(metadata, outfile)

    with open(os.path.join(output_data_dir, 'features.pkl'), 'wb') as outfile:
        pickle.dump(all_features, outfile)



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
