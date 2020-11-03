import os
import csv
import nltk
nltk.download('punkt')

from argparse import ArgumentParser
from torchtext.vocab import GloVe
from progress.bar import Bar

# Assert required parameters
def run_assertions(args):
    assert os.path.isdir(args.audio_data_dir), f"{args.audio_data_dir} does not exist."
    assert os.path.isdir(args.text_data_dir), f"{args.text_data_dir} does not exist."

# Str to float middle function
# --undefined-- is set to a numerical value
def str_to_float(str_val, fileNum):
    if str_val == '--undefined--':
        print(f'{fileNum} chucked undefined!')
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
    glove = GloVe(name='twitter.27B', dim=200)
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

            cases.append([case, rows])

    return cases

# Extract audio features given directory for audio data
def extract_audio_features(audio_data_dir):

    return None

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
