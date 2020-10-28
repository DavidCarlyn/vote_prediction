import parselmouth
import numpy as np
import os.path

def str_to_float(str_val):
    if str_val == '--undefined--':
        print("undefined!")
        return -1.0
    else:
        return float(str_val)

# Choose which year to run this for
year = "2015"
yearFolderPath = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/" + year + "/"

# Output file info
outputPath = "C:/Users/Webs/Documents/CSE/MachineLearn/Project/featureOutput/" + year

# Get the list of all the case folder paths.  Have to do this as the case numbers aren't contiguous.
# Also counting the number of case folders.
casePaths = []
for subdir, dirs, _ in os.walk(yearFolderPath):
    for dir in dirs:
        casePaths.append(os.path.join(subdir, dir))

# Loop over each case in the specified year
for casePath in casePaths:

    # Count case files in folder and preallocate a NumPy vector.  Doing this to avoid returning new arrays with .append().
    # Also declare a list for the feature values
    num_files = len([f for f in os.listdir(casePath) if os.path.isfile(os.path.join(casePath, f))])
    feature_matrix = np.zeros([num_files, 26], dtype='float32')

    fileIndex = 0
    print (casePath)

    # Loop over all the cases in the specified year.  The + 20 is to compensate for any missing files.
    # This is problemetic, as there might be more than 20 missing files from a case.  Come back to this.
    for file_number in range(num_files + 20):
        
        filePath = casePath + "/" + casePath[-10:] + "_" + str(file_number) + ".mp3"
        
        # These are apparently the 3 objects you need.  The try block is because some of the files are missing.
        try:
            sound = parselmouth.Sound(filePath)
        except:
            print(f'file {filePath} is missing!')
            continue

        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        voice_report_str = parselmouth.praat.call([sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)
        VPRsplit = voice_report_str.split()
        feature_list = []
    
        feature_list.append(str_to_float(VPRsplit[11]))
        feature_list.append(str_to_float(VPRsplit[15]))
        feature_list.append(str_to_float(VPRsplit[19]))
        feature_list.append(str_to_float(VPRsplit[23]))
        feature_list.append(str_to_float(VPRsplit[27]))
    
        feature_list.append(str_to_float(VPRsplit[33]))
        feature_list.append(str_to_float(VPRsplit[37]))
        feature_list.append(str_to_float(VPRsplit[40]))
        feature_list.append(str_to_float(VPRsplit[46]))
    
        feature_list.append(str_to_float(VPRsplit[54].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[62]))
        feature_list.append(str_to_float(VPRsplit[67].strip('%')) / 100.0)
    
        feature_list.append(str_to_float(VPRsplit[76].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[80]))
        feature_list.append(str_to_float(VPRsplit[84].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[87].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[90].strip('%')) / 100.0)
    
        feature_list.append(str_to_float(VPRsplit[94].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[98]))
        feature_list.append(str_to_float(VPRsplit[102].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[105].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[108].strip('%')) / 100.0)
        feature_list.append(str_to_float(VPRsplit[111].strip('%')) / 100.0)
    
        feature_list.append(str_to_float(VPRsplit[120]))
        feature_list.append(str_to_float(VPRsplit[124]))
        feature_list.append(str_to_float(VPRsplit[128]))

        # Copy the feature list into the correct cells of the numpy array
        for i in range(26):
            feature_matrix[fileIndex, i] = feature_list[i]

        fileIndex = fileIndex + 1
        
        #print (f'file {filePath} completed!')

    outputFile = outputPath + "/" + casePath[-10:] + ".txt"
    np.savetxt(outputFile, feature_matrix)
    
    print (casePath + " completed!")
    break