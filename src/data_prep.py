import os
import csv
import json
import parselmouth
import nltk
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

# Extract text features given directory for textual data
# Header rows are skipped (so starting rows will have # = 2)
# If a row can't be read, only the row # and a None value will be in place there [row_idx, None]
# If a row can be read, but the sentence/word can't be tokenized, the last value will be None (for no features)
# If a row can be read, and the sentence is tokenizable, then the row will be as follows:
# [row_idx, speaker_ID, speaker_role, start, stop, byte_start, byte_stop, [[word_embeddings], [word_embeddings], ...words..., [word_embeddings]]]
def extract_text_features(text_data_dir):
    cases = []
    glove = GloVe(name='twitter.27B', dim=TEXT_FEATURES)
    for root, dirs, files in os.walk(text_data_dir):
        for f in Bar(f"Reading files from {root}").iter(files):
            case = f.split('_')[1]
            case = case.split('.')[0]

            rows = []
            with open(os.path.join(root, f)) as csv_file:
                reader = csv.reader(csv_file)
                i = 0

                row = 0
                while row is not None:
                    try:
                        i += 1
                        row = next(reader, None)
                        if i == 1 or row == None: continue
                        try:
                            sentence = row[-1]
                            tokens = nltk.tokenize.word_tokenize(sentence)
                            embeddings = []
                            for t in tokens:
                                embeddings.append(glove[t])
                            row[-1] = embeddings
                        except:
                            print(f"Error. Can't tokenize: {sentence}")
                            row[-1] = None
                        finally:
                            if i != 1:
                                row.insert(0, i)
                                rows.append(row)
                    except:
                        print(f"Can't read row {i} from {os.path.join(root, f)}")
                        if i != 1:
                            rows.append([i, None])

            cases.append([int(case), rows])
        #TMP
        return cases

    return cases

# Extract audio features given directory for audio data
def extract_audio_features(audio_data_dir):
    audio_map = {}
    for root, dirs, files in os.walk(audio_data_dir):
        for f in files:
            year, case, sentence = f.split('_')
            sentence, ext = sentence.split('.')
            if year not in audio_map:
                audio_map[year] = {}
            if case not in audio_map[year]:
                audio_map[year][case] = []
            audio_map[year][case].append([int(sentence), os.path.join(root, f)])

    cases = []
    broken_trans_files = []
    more_audio = []
    missing_files = []
    undefined_audio_files = []
    for year in audio_map:
        for case in audio_map[year]:
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
                        continue
                except:
                    print(f'{path} is missing!')
                    missing_files.append(path)
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
            #TEMP
            return cases
                
    return cases

def combine_features(text_features, audio_features):
    text_features.sort(key=lambda x: x[0])
    audio_features.sort(key=lambda x: x[0])
    metadata = []
    all_features = []
    done = False
    text_i = 0
    audio_i = 0
    while not done:
        if text_i >= len(text_features) and audio_features < len(audio_features):
            metadata.append({ "case" : audio_features[audio_i][1][0], "valid" : False })
            all_features.append(None)
            audio_i += 1
            continue
        elif text_i < len(text_features) and audio_features >= len(audio_features):
            metadata.append({ "case" : test_features[text_i][1][0], "valid" : False })
            all_features.append(None)
            text_i += 1
            continue
        elif text_i >= len(text_features) and audio_features >= len(audio_features):
            done = True
            continue

        case_audio_features = audio_features[audio_i][1]
        case_text_features = text_features[text_i][1]
        text_case = case_text_features[0]
        audio_case = case_audio_features[0]
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
                    sen_meta.append({"sentence_num" : case_audio_features[sen_j], "valid" : False})
                    sen_j += 1
                    continue
                elif sen_i < len(case_text_features) and sen_j >= len(case_audio_features):
                    sen_features.append(None)
                    sen_meta.append({"sentence_num" : case_text_features[sen_i], "valid" : False})
                    sen_i += 1
                    continue
                elif sen_i >= len(case_text_features) and sen_j >= len(case_audio_features):
                    sen_done = True
                    continue

                text_sen = case_text_features[sen_i]
                audio_sen = case_audio_features[sen_j]
                if text_sen[0]-1 != audio_sen[0]:
                    sen_features.append(None)
                    if text_sen[0]-1 < audio_sen[0]:
                        sen_meta.append({ "sentence_num" : text_sen, "valid" : False })
                        sen_i += 1
                    else:
                        sen_meta.append({ "sentence_num" : audio_sen, "valid" : False })
                        sen_j += 1
                else:
                    combined = audio_sen[-1]
                    combined.extend(text_sen[-1])
                    sen_features.append(combined)
                    sen_meta.append({
                        "valid" : True,
                        "sentence_num" : audio_sen[1][0],
                        "speaker_id" : text_sen[1][1],
                        "speaker_role" : text_sen[1][2]
                    })

                    sen_i += 1
                    sen_j += 1

            metadata.append(sen_meta)
            all_features.append(sen_features)

            text_i += 1
            audio_i += 1

    return metadata, all_features


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

    # Extracting text features
    text_features = extract_text_features(args.text_data_dir)

    # Extracting audio features
    audio_features = extract_audio_features(args.audio_data_dir)

    # Combine features
    metadata, features = combine_features(text_features, audio_features)

    # Save Results
    with open('metadata.json', 'w') as outfile:
        json.dump(metadata, outfile)

    with open('features.json', 'w') as outfile:
        json.dump(features, outfile)
