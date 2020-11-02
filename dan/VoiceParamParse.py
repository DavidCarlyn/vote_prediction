import parselmouth
import numpy as np
import os.path
import glob
import csv
from pathlib import Path
from multiprocessing import Pool

def str_to_float(str_val, fileNum):
    if str_val == '--undefined--':
        print(f'{fileNum} chucked undefined!')
        return -1.0
    else:
        return float(str_val)

def voice_param_parse(year):
    yearFolderPath = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/" + year + "/"
    transcriptFolderPath = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/transcript/"

    # Output file info
    outputPath = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/featureOutput/" + year

    # Get the list of all the case folder paths.  Have to do this as the case numbers aren't contiguous.
    casePaths = []
    for subdir, dirs, _ in os.walk(yearFolderPath):
        for dir in dirs:
            casePaths.append(os.path.join(subdir, dir))

    # Loop over each case in the specified year.  I'm recording any broken files or cases where number of audio files > trans lines
    broken_trans_files = []
    more_audio = []
    for casePath in casePaths:

        # Count case files in folder and preallocate a NumPy vector.  Doing this to avoid returning new arrays with .append().
        # Also declare a list for the feature values
        num_audio_files = len([f for f in os.listdir(casePath) if os.path.isfile(os.path.join(casePath, f))])

        # This part is all for counting the lines in the transcript
        lines = -1
        case_number = casePath[-5:]
        transcriptPath = glob.glob(transcriptFolderPath + "*" + case_number + ".csv")[0]
        try:
            file = open(transcriptPath)
            reader = csv.reader(file)
            lines = len(list(reader))
        except:
            broken_trans_files.append(case_number)

        if (0 < lines < num_audio_files):
            more_audio.append(case_number)

        matrix_rows = max(num_audio_files, lines)
        feature_matrix = np.zeros([matrix_rows, 26], dtype='float32')

        fileIndex = 0
        print(casePath)

        # Loop over all the cases in the specified year.
        missingFiles = []
        undefineds = []
        for file_number in range(matrix_rows):
        
            caseNumber = casePath[-10:]
            filePath = casePath + "/" + caseNumber + "_" + str(file_number) + ".mp3"
        
            # These are apparently the 3 objects you need.  The try block is because some of the files are missing.
            # The size check is to dodge small, empty, corrupt files that were breaking even the try block.
            try:
                fileSize = Path(filePath).stat().st_size
                if fileSize > 2000:
                    sound = parselmouth.Sound(filePath)
                else:
                    print(f'Deleting {filePath}!')
                    missingFiles.append(file_number)
                    os.remove(filepath)
                    continue
            except:
                print(f'{filePath} is missing!')
                missingFiles.append(file_number)
                continue

            pitch = sound.to_pitch()
            pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

            voice_report_str = parselmouth.praat.call([sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)
            VPRsplit = voice_report_str.split()
            feature_list = []
    
            feature_list.append(str_to_float(VPRsplit[11], file_number))
            feature_list.append(str_to_float(VPRsplit[15], file_number))
            feature_list.append(str_to_float(VPRsplit[19], file_number))
            feature_list.append(str_to_float(VPRsplit[23], file_number))
            feature_list.append(str_to_float(VPRsplit[27], file_number))
    
            feature_list.append(str_to_float(VPRsplit[33], file_number))
            feature_list.append(str_to_float(VPRsplit[37], file_number))
            feature_list.append(str_to_float(VPRsplit[40], file_number))
            feature_list.append(str_to_float(VPRsplit[46], file_number))
    
            feature_list.append(str_to_float(VPRsplit[54].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[62], file_number))
            feature_list.append(str_to_float(VPRsplit[67].strip('%'), file_number) / 100.0)
    
            feature_list.append(str_to_float(VPRsplit[76].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[80], file_number))
            feature_list.append(str_to_float(VPRsplit[84].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[87].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[90].strip('%'), file_number) / 100.0)
    
            feature_list.append(str_to_float(VPRsplit[94].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[98], file_number))
            feature_list.append(str_to_float(VPRsplit[102].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[105].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[108].strip('%'), file_number) / 100.0)
            feature_list.append(str_to_float(VPRsplit[111].strip('%'), file_number) / 100.0)
    
            feature_list.append(str_to_float(VPRsplit[120], file_number))
            feature_list.append(str_to_float(VPRsplit[124], file_number))
            feature_list.append(str_to_float(VPRsplit[128], file_number))

            if -1.0 in feature_list:
                undefineds.append((caseNumber, file_number))

            # Copy the feature list into the correct cells of the numpy array
            for i in range(26):
                feature_matrix[fileIndex, i] = feature_list[i]

            fileIndex = fileIndex + 1

        outputFile = outputPath + "/" + casePath[-10:] + ".txt"
        with open(outputFile, "a") as f:
            f.write("%s\n" % missingFiles)
            np.savetxt(f, feature_matrix)
            f.close()
    
        print(casePath + " completed!")
        print()

    broken_trans_outfile = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/featureOutput/" + year + "/brokenTransFiles.txt"
    btf = open(broken_trans_outfile, "w")
    btf.write("%s" % broken_trans_files)

    moreAudioOut = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/featureOutput/" + year + "/moreAudio.txt"
    mao = open(moreAudioOut, "w")
    mao.write("%s" % more_audio)

    undefinedsOut = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/featureOutput/" + year + "/undefineds.txt"
    uno = open(undefinedsOut, "w")
    uno.write("%s" % undefineds)

    btf.close()
    mao.close()
    uno.close()

# Driver code to parse all the years using multiple processes
if __name__ == '__main__':
    years = [str(i) for i in range(2001, 2016)]
    with Pool(3) as p:
        p.map(voice_param_parse, years)